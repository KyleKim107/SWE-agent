"""
Run on a batch of instances/issues, e.g., SWE-bench.

[cyan][bold]=== BASIC OPTIONS ===[/bold][/cyan]

  -h --help           Show help text and exit
  --help_option      Print specific help text and exit

[cyan][bold]=== EXAMPLES ===[/bold][/cyan]

Basic usage: Run over a [bold][cyan]SWE-bench lite[/bold][/cyan][green]:

sweagent run-batch \\
    --instances.type swe_bench \\ # configure instances
    --instances.subset lite \\
    --instances.split dev  \\
    --instances.slice :50 \\     # first 50 instances
    --instances.shuffle=True \\  # shuffle instances (with fixed seed)
    --config config/default.yaml \\
    --agent.model.name gpt-4o  # configure model
[/green]

[cyan][bold]=== LOADING INSTANCES ===[/bold][/cyan]

[cyan][bold]From a file[/bold][/cyan] [green]--instances.type file --instances.path /path/to/file[/green].
[cyan][bold]From huggingface[/bold][/cyan] [green]--instances.type huggingface --instances.dataset_name=SWE_Bench_lite --instances.split=dev[/green].

All instance specifications support the [green]filter[/green], [green]slice[/green], and [green]shuffle[/green] options.
With [green]filter[/green], you can select specific instances, e.g., [green]--instances.filter='instance_id_1|instance_id_2'[/green].
"""

import getpass
import json
import logging
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.live import Live
from swerex.deployment.hooks.status import SetStatusDeploymentHook

from sweagent import TRAJECTORY_DIR
from sweagent.agent.agents import AgentConfig, get_agent_from_config
from sweagent.agent.hooks.status import SetStatusAgentHook
from sweagent.environment.hooks.status import SetStatusEnvironmentHook
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import ModelConfigurationError, TotalCostLimitExceededError
from sweagent.run._progress import RunBatchProgressManager
from sweagent.run.batch_instances import BatchInstance, BatchInstanceSourceConfig, SWEBenchInstances
from sweagent.run.common import BasicCLI, ConfigHelper, save_predictions
from sweagent.run.hooks.abstract import CombinedRunHooks, RunHook
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.merge_predictions import merge_predictions
from sweagent.run.run_single import RunSingleConfig
from sweagent.types import AgentRunResult
from sweagent.utils.config import load_environment_variables
from sweagent.utils.log import (
    add_file_handler,
    add_logger_names_to_stream_handlers,
    get_logger,
    register_thread_name,
    remove_file_handler,
    set_stream_handler_levels,
)


class RunBatchConfig(BaseSettings, cli_implicit_flags=False):
    instances: BatchInstanceSourceConfig = Field(description="Instances to run.")
    agent: AgentConfig = Field(description="Agent options.")
    output_dir: Path = Field(default=Path("DEFAULT"), description="Output directory.")
    suffix: str = ""
    """Suffix to add to the output directory. Only used if `output_dir` is `DEFAULT`."""
    raise_exceptions: bool = False
    """Raise exceptions instead of skipping instances."""
    redo_existing: bool = False
    """Do not skip instances that already have a trajectory."""
    env_var_path: Path | None = None
    """Path to a .env file to load environment variables from."""
    num_workers: int = Field(default=1)
    """Number of parallel workers to use."""
    random_delay_multiplier: float = 0.3
    """We will wait for a random amount of time between 0 and `random_delay_multiplier`
    times the number of workers at the start of each instance. This is to avoid any
    potential race condition or issues with bottlenecks, e.g., when running on a platform
    with few CPUs that cannot handle the startup of all containers in time.
    """
    progress_bar: bool = True
    """Whether to show a progress bar. Progress bar is never shown for human models.
    Progress bar is always shown for multi-worker runs.
    """

    # pydantic config
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SWE_AGENT_")

    def set_default_output_dir(self) -> None:
        # Needs to be called explicitly, because self._config_files will be setup
        # post-init.
        if self.output_dir == Path("DEFAULT"):
            user_id = getpass.getuser()
            source_id = self.instances.id
            try:
                model_id = self.agent.model.id  # type: ignore[attr-defined]
            except AttributeError:
                model_id = "unknown"
            config_file = getattr(self, "_config_files", ["no_config"])[0]
            if config_file != "no_config":
                config_file = Path(config_file).stem
            suffix = f"__{self.suffix}" if self.suffix else ""
            self.output_dir = TRAJECTORY_DIR / user_id / f"{config_file}__{model_id}___{source_id}{suffix}"

    @model_validator(mode="after")
    def evaluate_and_redo_existing(self) -> Self:
        if not isinstance(self.instances, SWEBenchInstances):
            return self
        if self.instances.evaluate and self.redo_existing:
            msg = (
                "Cannot evaluate and redo existing at the same time. This would cause invalid results, because "
                "after the first merge_preds gives you a preds.json, this file would be submitted to SB-CLI, causing"
                "evaluation of old instances, which could then not be overwritten by the new ones."
            )
            raise ValueError(msg)
        return self


class _BreakLoop(Exception):
    """Used for internal control flow"""


class RunBatch:
    def __init__(
        self,
        instances: list[BatchInstance],
        agent_config: AgentConfig,
        *,
        output_dir: Path = Path("."),
        hooks: list[RunHook] | None = None,
        raise_exceptions: bool = False,
        redo_existing: bool = False,
        num_workers: int = 1,
        progress_bar: bool = True,
        random_delay_multiplier: float = 0.3,
    ):
        """Note: When initializing this class, make sure to add the hooks that are required by your actions.
        See `from_config` for an example.

        Args:
            hooks: If not specified, the default hooks will be used.
            num_workers: Number of parallel workers to use. Default is 1 (sequential execution).
            progress_bar: Whether to show a progress bar. Progress bar is never shown for human models.
                Progress bar is always shown for multi-worker runs.
            random_delay_multiplier: We will wait for a random amount of time between 0 and `random_delay_multiplier`
                times the number of workers at the start of each instance. This is to avoid any
                potential race conditions.
        """
        if self._model_id in ["human", "human_thought"] and num_workers > 1:
            msg = "Cannot run with human model in parallel"
            raise ValueError(msg)

        self.logger = get_logger("swea-run", emoji="🏃")
        add_file_handler(
            output_dir / "run_batch.log",
            id_="progress",
            filter=lambda name: "swea-run" in name or "config" in name,
        )
        self.instances = instances
        self.agent_config = agent_config
        self.output_dir = output_dir
        self._raise_exceptions = raise_exceptions
        self._chooks = CombinedRunHooks()
        self._redo_existing = redo_existing
        self._num_workers = min(num_workers, len(instances))
        for hook in hooks or [SaveApplyPatchHook(show_success_message=False)]:
            self.add_hook(hook)
        self._progress_manager = RunBatchProgressManager(
            num_instances=len(instances), yaml_report_path=output_dir / "run_batch_exit_statuses.yaml"
        )
        self._show_progress_bar = progress_bar
        self._random_delay_multiplier = random_delay_multiplier

    @property
    def _model_id(self) -> str:
        try:
            return self.agent_config.model.id  # type: ignore[attr-defined]
        except AttributeError:
            return "unknown"

    @classmethod
    def from_config(cls, config: RunBatchConfig) -> Self:
        load_environment_variables(config.env_var_path)
        config.set_default_output_dir()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "run_batch.config.yaml").write_text(yaml.dump(config.model_dump_json(), indent=2))
        logger = get_logger("run", emoji="🏃")
        logger.debug("Loading instances from %s", f"{config.instances!r}")
        instances = config.instances.get_instance_configs()
        logger.info("Loaded %d instances", len(instances))
        if not instances:
            msg = (
                "No instances to run. Here are a few things to check:\n"
                "- With huggingface data: Check that you have the right split (test or dev)\n"
                "- Check your filter does not exclude all instances (check the info log messages)"
            )
            raise ValueError(msg)
        logger.debug("The first instance is %s", f"{instances[0]!r}")
        rb = cls(
            instances=instances,
            agent_config=config.agent,
            output_dir=config.output_dir,
            raise_exceptions=config.raise_exceptions,
            redo_existing=config.redo_existing,
            num_workers=config.num_workers,
            progress_bar=config.progress_bar,
            random_delay_multiplier=config.random_delay_multiplier,
        )
        if isinstance(config.instances, SWEBenchInstances) and config.instances.evaluate:
            from sweagent.run.hooks.swe_bench_evaluate import SweBenchEvaluate

            rb.add_hook(
                SweBenchEvaluate(
                    output_dir=config.output_dir,
                    subset=config.instances.subset,
                    split=config.instances.split,
                    continuous_submission_every=30,
                )
            )
        return rb

    def add_hook(self, hook: RunHook) -> None:
        hook.on_init(run=self)
        self._chooks.add_hook(hook)

    def main(self) -> None:
        self.logger.info("Starting run. Find output files at %s", self.output_dir)
        self._chooks.on_start()

        if self._num_workers <= 1:
            self.main_single_worker()
        else:
            self.main_multi_worker()

        output_dirs = []
        for instance in self.instances:
            output_dirs.append(self.output_dir / instance.problem_statement.id)
        merge_predictions(output_dirs, self.output_dir / "preds.json")

        self._chooks.on_end()

    def main_single_worker(self) -> None:
        with ExitStack() as stack:
            # Conditionally add progress bar
            if self._model_id not in ["human", "human_thought"] and self._show_progress_bar:
                stack.enter_context(Live(self._progress_manager.render_group))
            for instance in self.instances:
                try:
                    self.run_instance(instance)
                except _BreakLoop:
                    self.logger.info("Stopping loop over instances")
                    break

    def main_multi_worker(self) -> None:
        add_logger_names_to_stream_handlers()
        # Set all stream handlers to WARNING and set everything where we want to have
        # more verbosity explicitly
        set_stream_handler_levels(logging.WARNING)
        self.logger.setLevel(logging.TRACE)  # type: ignore

        with Live(self._progress_manager.render_group):
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                futures = [executor.submit(self.run_instance, instance) for instance in self.instances]
                try:
                    for future in as_completed(futures):
                        future.result()
                except (KeyboardInterrupt, _BreakLoop):
                    msg = (
                        "Received keyboard interrupt, waiting for running instances "
                        "to finish, but cancelled everything else"
                    )
                    self.logger.info(msg)
                    executor.shutdown(wait=False, cancel_futures=True)
                finally:
                    self._progress_manager.print_report()

    def run_instance(self, instance: BatchInstance) -> None:
        self.logger.info("Running on instance %s", instance.problem_statement.id)
        register_thread_name(instance.problem_statement.id)
        self._add_instance_log_file_handlers(instance.problem_statement.id, multi_worker=self._num_workers > 1)
        # Let's add some randomness to avoid any potential race conditions or thundering herd
        if self._progress_manager.n_completed < self._num_workers:
            time.sleep(random.random() * self._random_delay_multiplier * (self._num_workers - 1))

        self._progress_manager.on_instance_start(instance.problem_statement.id)

        if previous_exit_status := self.should_skip(instance):
            self._progress_manager.on_instance_end(
                instance.problem_statement.id, exit_status=f"skipped ({previous_exit_status})"
            )
            self._remove_instance_log_file_handlers(instance.problem_statement.id)
            return

        # Either catch and silence exception, or raise _BreakLoop to stop the loop
        # over the instances
        try:
            result = self._run_instance(instance)
        except KeyboardInterrupt:
            raise _BreakLoop
        except (SystemExit, ModelConfigurationError, TotalCostLimitExceededError) as e:
            if self._raise_exceptions:
                raise
            self.logger.critical(f"❌ Exiting because {e.__class__.__name__} was called")
            raise _BreakLoop
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"❌ Failed on {instance.problem_statement.id}: {e}")
            self._progress_manager.on_uncaught_exception(instance.problem_statement.id, e)
            if self._raise_exceptions:
                raise
        else:
            self._progress_manager.on_instance_end(
                instance.problem_statement.id, exit_status=result.info.get("exit_status", "unknown_exit")
            )
        finally:
            self._progress_manager.update_exit_status_table()
            self._remove_instance_log_file_handlers(instance.problem_statement.id)

    def _run_instance(self, instance: BatchInstance) -> AgentRunResult:
        output_dir = Path(self.output_dir) / instance.problem_statement.id
        output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_config.name = f"{instance.problem_statement.id}"
        agent = get_agent_from_config(self.agent_config)
        single_run_replay_config = RunSingleConfig(
            agent=self.agent_config,
            problem_statement=instance.problem_statement,
            env=instance.env,
        )
        (output_dir / f"{instance.problem_statement.id}.config.yaml").write_text(
            yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
        )
        agent.replay_config = single_run_replay_config  # type: ignore[attr-defined]
        agent.add_hook(SetStatusAgentHook(instance.problem_statement.id, self._progress_manager.update_instance_status))
        self._progress_manager.update_instance_status(instance.problem_statement.id, "Starting environment")
        instance.env.name = f"{instance.problem_statement.id}"
        env = SWEEnv.from_config(instance.env)
        env.add_hook(
            SetStatusEnvironmentHook(instance.problem_statement.id, self._progress_manager.update_instance_status)
        )
        env.deployment.add_hook(
            SetStatusDeploymentHook(instance.problem_statement.id, self._progress_manager.update_instance_status)
        )
        try:
            env.start()
            self._chooks.on_instance_start(index=0, env=env, problem_statement=instance.problem_statement)
            result = agent.run(
                problem_statement=instance.problem_statement,
                env=env,
                output_dir=output_dir,
            )
        except Exception:
            # The actual handling is happening in `run_instance`, but we need to make sure that
            # we log it to the agent specific logger as well
            agent.logger.error(traceback.format_exc())  # type: ignore[attr-defined]
            raise
        finally:
            container_info = self._get_container_cleanup_info(env)
            env.close()
            # Wait a moment for container to fully stop (especially if --rm is used)
            time.sleep(0.5)
            # Clean up Docker container and image after instance completion
            self._cleanup_docker_container(container_info)
            self._cleanup_docker_image(instance, env)
        save_predictions(self.output_dir, instance.problem_statement.id, result)
        self._chooks.on_instance_completed(result=result)
        return result

    def _get_container_cleanup_info(self, env: SWEEnv) -> tuple[str | None, str | None, bool]:
        """Capture container metadata before teardown."""
        deployment = getattr(env, "deployment", None)
        if deployment is None:
            return (None, None, False)
        config = getattr(deployment, "_config", None)
        container_runtime = getattr(config, "container_runtime", None)
        remove_container = bool(getattr(config, "remove_container", False))
        container_name = getattr(deployment, "container_name", None)
        return (container_name, container_runtime, remove_container)

    def _cleanup_docker_container(self, container_info: tuple[str | None, str | None, bool]) -> None:
        """Remove Docker container after an instance finishes."""
        container_name, container_runtime, remove_container = container_info
        if not remove_container or not container_name:
            return

        runtime = container_runtime or "docker"
        try:
            # Check if container exists first
            inspect_result = subprocess.run(
                [runtime, "inspect", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if inspect_result.returncode != 0:
                # Container doesn't exist (might have been auto-removed by --rm)
                self.logger.debug(f"Container {container_name} already removed")
                return
            
            # First, try to stop the container if it's still running
            stop_result = subprocess.run(
                [runtime, "stop", container_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if stop_result.returncode == 0:
                self.logger.debug(f"Stopped Docker container: {container_name}")
            # Container might already be stopped, which is fine
            
            # Now remove the container (force removal in case it's still running)
            result = subprocess.run(
                [runtime, "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                self.logger.info(f"✅ Removed Docker container: {container_name}")
            else:
                # Check if container doesn't exist (might have been auto-removed by --rm)
                if "No such container" in result.stderr or "no such container" in result.stderr.lower():
                    self.logger.debug(f"Container {container_name} already removed (likely by --rm flag)")
                else:
                    self.logger.warning(
                        f"Could not remove Docker container {container_name} via {runtime}: {result.stderr.strip()}"
                    )
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout removing Docker container {container_name}")
        except Exception as e:
            self.logger.warning(f"Error removing Docker container {container_name}: {e}")

    def _cleanup_docker_image(self, instance: BatchInstance, env: SWEEnv) -> None:
        """Remove Docker image after instance completion to free up disk space."""
        try:
            # Get image name and container runtime from deployment config
            deployment = getattr(env, "deployment", None)
            if deployment is None:
                self.logger.debug("No deployment found for image cleanup")
                return
            
            config = getattr(deployment, "_config", None)
            if config is None:
                self.logger.debug("No deployment config found for image cleanup")
                return
            
            image_name = getattr(config, "image", None)
            container_runtime = getattr(config, "container_runtime", "docker")
            
            if not image_name:
                # Try to get from instance config as fallback
                instance_deployment = getattr(instance.env, "deployment", None)
                if instance_deployment:
                    if hasattr(instance_deployment, "image"):
                        image_name = instance_deployment.image
                    elif hasattr(instance_deployment, "_config"):
                        image_name = getattr(instance_deployment._config, "image", None)
            
            if not image_name:
                self.logger.debug("No image name found for cleanup")
                return
            
            runtime = container_runtime or "docker"
            
            # Only remove SWE-bench images (not base images like python:3.11)
            # This prevents accidentally removing shared base images
            image_lower = image_name.lower()
            if "swebench" in image_lower or "sweb.eval" in image_lower:
                self.logger.info(f"🖼️  Attempting to remove {runtime} image: {image_name}")
                
                # First check if image exists
                inspect_result = subprocess.run(
                    [runtime, "inspect", image_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                
                if inspect_result.returncode != 0:
                    # Image doesn't exist, might have been removed already or might be a built image
                    self.logger.debug(f"Image {image_name} not found, checking for related images")
                    # Try to find and remove any dangling/built images that might be related
                    self._cleanup_built_images(runtime, image_name)
                    # Also try to remove by repository pattern
                    self._cleanup_images_by_pattern(runtime, image_name)
                    return
                
                # Remove the image (force removal)
                result = subprocess.run(
                    [runtime, "rmi", "-f", image_name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self.logger.info(f"✅ Removed {runtime} image: {image_name}")
                    # Also try to clean up any dangling images that might have been built
                    self._cleanup_built_images(runtime, image_name)
                    # Clean up any other images matching the pattern
                    self._cleanup_images_by_pattern(runtime, image_name)
                else:
                    # Image might be in use by another container or have dependencies
                    error_msg = result.stderr.lower()
                    stderr_full = result.stderr.strip()
                    if "image is being used" in error_msg or "is being used by" in error_msg:
                        self.logger.warning(f"⚠️  Image {image_name} is still in use, cannot remove: {stderr_full}")
                        # Try to remove any containers using this image first
                        self._remove_containers_using_image(runtime, image_name)
                        # Try again after removing containers
                        retry_result = subprocess.run(
                            [runtime, "rmi", "-f", image_name],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if retry_result.returncode == 0:
                            self.logger.info(f"✅ Removed {runtime} image: {image_name} (after removing containers)")
                            # Clean up related images
                            self._cleanup_built_images(runtime, image_name)
                            self._cleanup_images_by_pattern(runtime, image_name)
                    elif "No such image" in result.stderr or "no such image" in error_msg:
                        self.logger.debug(f"Image {image_name} already removed")
                    else:
                        self.logger.warning(f"⚠️  Could not remove {runtime} image {image_name}: {stderr_full}")
            else:
                self.logger.debug(f"Skipping image cleanup for non-SWE-bench image: {image_name}")
        except Exception as e:
            # Don't fail the entire run if image cleanup fails
            self.logger.warning(f"Error cleaning up Docker image: {e}", exc_info=True)
    
    def _remove_containers_using_image(self, runtime: str, image_name: str) -> None:
        """Remove any containers that are using the specified image."""
        try:
            # Find containers using this image
            ps_result = subprocess.run(
                [runtime, "ps", "-a", "--filter", f"ancestor={image_name}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if ps_result.returncode != 0:
                return
            
            container_ids = [cid.strip() for cid in ps_result.stdout.strip().split("\n") if cid.strip()]
            
            for container_id in container_ids:
                try:
                    # Force remove the container
                    rm_result = subprocess.run(
                        [runtime, "rm", "-f", container_id],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if rm_result.returncode == 0:
                        self.logger.debug(f"Removed container {container_id} that was using image {image_name}")
                except Exception:
                    pass  # Ignore errors
        except Exception as e:
            self.logger.debug(f"Error removing containers using image: {e}")
    
    def _cleanup_images_by_pattern(self, runtime: str, base_image_name: str) -> None:
        """Clean up images matching the SWE-bench pattern, focusing on the current instance's image."""
        try:
            # Extract the instance-specific part from the image name
            # e.g., from "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-6938:latest"
            # we want to match images with the same instance ID
            instance_pattern = None
            if "sweb.eval" in base_image_name.lower():
                # Extract the instance identifier (e.g., "astropy_1776_astropy-6938")
                parts = base_image_name.lower().split("sweb.eval.")
                if len(parts) > 1:
                    instance_part = parts[1].split(":")[0].split("/")[-1]
                    instance_pattern = instance_part
            
            # List all images
            list_result = subprocess.run(
                [runtime, "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if list_result.returncode != 0:
                return
            
            # Find and remove images matching the pattern
            lines = list_result.stdout.strip().split("\n")
            removed_count = 0
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 1:
                    image_ref = parts[0]  # Repository:Tag
                    
                    # Check if this image matches the SWE-bench pattern
                    if "swebench" in image_ref.lower() or "sweb.eval" in image_ref.lower():
                        # Skip the base image we're already trying to remove
                        if image_ref == base_image_name or image_ref == base_image_name.split(":")[0] + ":latest":
                            continue
                        
                        # If we have an instance pattern, only remove images matching that specific instance
                        # Otherwise, be more conservative and only remove if it's clearly related
                        if instance_pattern and instance_pattern in image_ref.lower():
                            # This is the same instance, safe to remove
                            pass
                        elif not instance_pattern:
                            # No specific pattern, skip to avoid removing unrelated images
                            continue
                        else:
                            # Different instance, skip it
                            continue
                        
                        # Check if image is in use by any containers
                        ps_result = subprocess.run(
                            [runtime, "ps", "-a", "--filter", f"ancestor={image_ref}", "--format", "{{.ID}}"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        
                        # Only remove if no containers are using it
                        if ps_result.returncode == 0 and not ps_result.stdout.strip():
                            try:
                                rm_result = subprocess.run(
                                    [runtime, "rmi", "-f", image_ref],
                                    capture_output=True,
                                    text=True,
                                    timeout=10,
                                )
                                if rm_result.returncode == 0:
                                    self.logger.debug(f"Removed related image: {image_ref}")
                                    removed_count += 1
                            except Exception:
                                pass  # Ignore errors
            
            if removed_count > 0:
                self.logger.info(f"✅ Removed {removed_count} related SWE-bench image(s)")
        except Exception as e:
            self.logger.debug(f"Error cleaning up images by pattern: {e}")
    
    def _cleanup_built_images(self, runtime: str, base_image_name: str) -> None:
        """Clean up any dangling or built images that might have been created."""
        try:
            # List all images and find ones that might be related (dangling or built from this base)
            list_result = subprocess.run(
                [runtime, "images", "--format", "{{.ID}}\t{{.Repository}}\t{{.Tag}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if list_result.returncode != 0:
                return
            
            # Look for dangling images (those with <none> as repository/tag)
            # These are often built images that weren't tagged
            lines = list_result.stdout.strip().split("\n")
            removed_count = 0
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    repo = parts[1] if len(parts) > 1 else ""
                    # Check if it's a dangling image (untagged)
                    if repo == "<none>" or (len(parts) > 2 and parts[2] == "<none>"):
                        image_id = parts[0]
                        # Try to remove dangling images (these are often built images)
                        try:
                            rm_result = subprocess.run(
                                [runtime, "rmi", "-f", image_id],
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )
                            if rm_result.returncode == 0:
                                self.logger.debug(f"Removed dangling image: {image_id}")
                                removed_count += 1
                        except Exception:
                            pass  # Ignore errors when removing dangling images
            
            if removed_count > 0:
                self.logger.info(f"✅ Removed {removed_count} dangling image(s)")
        except Exception as e:
            self.logger.debug(f"Error cleaning up built images: {e}")

    def should_skip(self, instance: BatchInstance) -> bool | str:
        """Check if we should skip this instance.
        Returns previous exit status if the instance should be skipped.
        """
        if self._redo_existing:
            return False

        # Check if there's an existing trajectory for this instance
        log_path = self.output_dir / instance.problem_statement.id / (instance.problem_statement.id + ".traj")
        if not log_path.exists():
            return False

        content = log_path.read_text()
        if not content.strip():
            self.logger.warning("Found empty trajectory: %s. Removing.", log_path)
            log_path.unlink()
            return False

        try:
            data = json.loads(content)
            # If the trajectory has no exit status, it's incomplete and we will redo it
            exit_status = data["info"].get("exit_status", None)
            if exit_status == "early_exit" or exit_status is None:
                self.logger.warning(f"Found existing trajectory with no exit status: {log_path}. Removing.")
                log_path.unlink()
                return False
        except Exception as e:
            self.logger.error(f"Failed to check existing trajectory: {log_path}: {e}. Removing.")
            # If we can't check the trajectory, we will redo it
            log_path.unlink()
            return False
        # otherwise, we will skip it
        self.logger.info(f"⏭️ Skipping existing trajectory: {log_path}")
        return exit_status

    def _add_instance_log_file_handlers(self, instance_id: str, multi_worker: bool = False) -> None:
        filename_template = f"{instance_id}.{{level}}.log"
        for level in ["trace", "debug", "info"]:
            filter = instance_id if multi_worker else ""
            add_file_handler(
                self.output_dir / instance_id / filename_template.format(level=level),
                filter=filter,
                level=level,
                id_=f"{instance_id}-{level}",
            )

    def _remove_instance_log_file_handlers(self, instance_id: str) -> None:
        for level in ["trace", "debug", "info"]:
            remove_file_handler(f"{instance_id}-{level}")


def run_from_config(config: RunBatchConfig):
    RunBatch.from_config(config).main()


def run_from_cli(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
    assert __doc__ is not None
    help_text = (  # type: ignore
        __doc__ + "\n[cyan][bold]=== ALL THE OPTIONS ===[/bold][/cyan]\n\n" + ConfigHelper().get_help(RunBatchConfig)
    )
    run_from_config(BasicCLI(RunBatchConfig, help_text=help_text).get_config(args))  # type: ignore


if __name__ == "__main__":
    run_from_cli()
