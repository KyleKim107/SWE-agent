from pathlib import Path
import subprocess
import sys
import json
from datetime import datetime


def submit_to_swe_bench(
    preds_path: Path,
    output_dir: Path,
    subset: str = "lite",
    split: str = "test",
    run_id: str = None,
    max_workers: int = 1,
    timeout: int = 900
) -> Path:
    """
    Run LOCAL evaluation using swebench docker harness
    (Replaces sb-cli submit)
    
    This function runs evaluation locally using Docker containers.
    Make sure Docker is running before calling this function.
    """
    # 1. 데이터셋 이름 매핑 (HuggingFace 이름 기준)
    subset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "multimodal": "princeton-nlp/SWE-bench_Multimodal"
    }
    
    if subset not in subset_map:
        # 기본값으로 전체 데이터셋 시도 혹은 에러 처리
        dataset_name = "princeton-nlp/SWE-bench" 
        if subset != "full":
             print(f"⚠️ Warning: Unknown subset '{subset}', defaulting to full SWE-bench")
    else:
        dataset_name = subset_map[subset]
    
    if run_id is None:
        run_id = f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # 2. swebench 패키지가 설치되어 있는지 확인
    try:
        import swebench
        use_direct_import = True
    except ImportError:
        use_direct_import = False
        print("⚠️  swebench package not found, using subprocess method")
    
    # 3. 로컬 평가 실행
    print(f"\n🚀 Running LOCAL evaluation (Docker)...")
    print(f"   Dataset: {dataset_name}")
    print(f"   Split: {split}")
    print(f"   Predictions: {preds_path}")
    print(f"   Run ID: {run_id}")
    print(f"   Max Workers: {max_workers}")
    print()
    
    if use_direct_import:
        # swebench 패키지를 직접 import해서 사용
        try:
            from swebench.harness.run_evaluation import main as run_evaluation_main
            import argparse
            
            # argparse Namespace 객체 생성
            args = argparse.Namespace(
                dataset_name=dataset_name,
                predictions_path=str(preds_path),
                split=split,
                run_id=run_id,
                max_workers=max_workers,
                timeout=timeout,
                output_dir=str(output_dir)
            )
            
            # swebench의 main 함수 실행
            # swebench는 sys.argv를 사용할 수 있으므로, 임시로 설정
            old_argv = sys.argv
            try:
                sys.argv = [
                    "swebench.harness.run_evaluation",
                    "-d", dataset_name,
                    "-p", str(preds_path),
                    "-s", split,
                    "-id", run_id,
                    "--max_workers", str(max_workers),
                    "-t", str(timeout),
                    "--report_dir", str(output_dir)
                ]
                run_evaluation_main()
            finally:
                sys.argv = old_argv
                
        except Exception as e:
            print(f"⚠️  Direct import failed: {e}")
            print("   Falling back to subprocess method...")
            use_direct_import = False
    
    if not use_direct_import:
        # subprocess를 사용한 방법 (더 안정적)
        # swebench.harness.run_evaluation을 모듈로 실행
        # 참고: swebench는 -d, -p, -s, -id, -t 등의 짧은 형식과 --report_dir를 사용
        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "-d", dataset_name,  # --dataset_name 대신 -d
            "-p", str(preds_path),  # --predictions_path 대신 -p (필수)
            "-s", split,  # --split 대신 -s
            "-id", run_id,  # --run_id 대신 -id (필수)
            "--max_workers", str(max_workers),
            "-t", str(timeout),  # --timeout 대신 -t
            "--report_dir", str(output_dir)  # --output_dir 대신 --report_dir
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            # 실행 (시간이 좀 걸립니다)
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=False  # 출력을 실시간으로 보기 위해
            )
            print(f"   ✅ Evaluation complete")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Evaluation failed with exit code {e.returncode}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"   Error output: {e.stderr}")
            raise
        except FileNotFoundError:
            print(f"   ❌ Error: swebench package not found")
            print(f"   Please install swebench: pip install swebench")
            print(f"   Or install sb-cli: pip install git+https://github.com/SWE-bench/sb-cli.git")
            raise

    # 4. 결과 파일 찾기 및 이동
    # swebench harness는 여러 위치에 결과를 저장할 수 있습니다
    possible_locations = [
        output_dir / f"{run_id}.json",
        output_dir / "results.json",
        Path(f"{run_id}.json"),  # 현재 디렉토리
        preds_path.parent / f"{run_id}.json",
        preds_path.parent / "results.json",
    ]
    
    final_results = output_dir / "results.json"
    found_result = None
    
    for location in possible_locations:
        if location.exists() and location.is_file():
            found_result = location
            break
    
    if found_result and found_result != final_results:
        # 결과 파일을 output_dir로 이동
        if final_results.exists():
            final_results.unlink()  # 기존 파일 삭제
        found_result.rename(final_results)
        print(f"   📋 Results moved to: {final_results}")
    elif found_result:
        print(f"   📋 Results found at: {final_results}")
    else:
        # 결과 파일을 찾지 못한 경우, output_dir에서 최근 json 파일 찾기
        json_files = list(output_dir.glob("*.json"))
        if json_files:
            # preds.json을 제외하고 가장 최근 파일 찾기
            result_files = [f for f in json_files if f.name != "preds.json"]
            if result_files:
                # 수정 시간 기준으로 정렬
                result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                found_result = result_files[0]
                if found_result != final_results:
                    if final_results.exists():
                        final_results.unlink()
                    found_result.rename(final_results)
                    print(f"   📋 Results found and moved to: {final_results}")
                else:
                    print(f"   📋 Results found at: {final_results}")
            else:
                print(f"   ⚠️  Could not find evaluation result file.")
                print(f"   Please check the output directory: {output_dir}")
        else:
            print(f"   ⚠️  Could not find evaluation result file.")
            print(f"   Please check the output directory: {output_dir}")
            # 에러를 내지 않고 빈 결과 파일 생성
            final_results.write_text(json.dumps({"error": "Results file not found"}, indent=2))
    
    return final_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SWE-bench results using local Docker harness")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory containing preds.json")
    parser.add_argument("--subset", type=str, default="lite", choices=["lite", "verified", "multimodal", "full"], help="SWE-bench subset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run ID for evaluation")
    parser.add_argument("--config", type=str, default=None, help="Config file (optional, not used by this script)")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers (default: 1, recommended for local Docker)")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per instance in seconds (default: 900)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    preds_path = output_dir / "preds.json"
    
    if not preds_path.exists():
        print(f"❌ Error: predictions file not found at {preds_path}")
        print(f"   Please ensure preds.json exists in {output_dir}")
        exit(1)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Evaluating SWE-bench results (Local Docker)...")
    print(f"   Predictions: {preds_path}")
    print(f"   Output dir: {output_dir}")
    print(f"   Subset: {args.subset}")
    print(f"   Split: {args.split}")
    print(f"   Max Workers: {args.max_workers}")
    print(f"   Timeout: {args.timeout}s per instance")
    print()
    
    # Docker가 실행 중인지 확인
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print("⚠️  Warning: Docker may not be running. Please start Docker before evaluation.")
            print()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  Warning: Docker not found or not accessible. Please ensure Docker is installed and running.")
        print()
    
    try:
        result_path = submit_to_swe_bench(
            preds_path=preds_path,
            output_dir=output_dir,
            subset=args.subset,
            split=args.split,
            run_id=args.run_id,
            max_workers=args.max_workers,
            timeout=args.timeout
        )
        print(f"\n✅ Evaluation complete! Results saved to: {result_path}")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)