# SWE-Agent TAMU API 개선 가이드

## 🎯 개선 사항

### 1. Temperature 조정
- **이전**: 0.7 (창의적이지만 불안정)
- **현재**: 0.0 (일관적이고 예측 가능)
- **효과**: Format error와 syntax error 감소

### 2. 강화된 프롬프트
- ✅ 명확한 출력 형식 지시
- ✅ 좋은/나쁜 명령어 예제 제공
- ✅ Bash escaping 주의사항 추가
- ✅ str_replace_editor 사용 권장

### 3. LLM 출력 가시화
- 📤 모든 LLM 출력이 로그에 표시됨
- 🔍 디버깅이 훨씬 쉬워짐

## 🚀 사용 방법

### 기본 실행 (개선된 config 사용)
```bash
cd /Users/kylekim/Desktop/TAMU/CS689/course_project/SWE-agent

sweagent run \
  --config config/tamu_config_improved.yaml \
  --agent.model.name "openai/protected.gpt-5" \
  --env.repo.github_url "https://github.com/django/django" \
  --problem_statement.github_url "https://github.com/django/django/issues/11099"
```

### Claude 모델 사용 (추천)
```bash
sweagent run \
  --config config/tamu_config_improved.yaml \
  --agent.model.name "openai/protected.Claude Sonnet 4.5" \
  --env.repo.github_url "https://github.com/django/django" \
  --problem_statement.github_url "https://github.com/django/django/issues/11099"
```

### 더 낮은 Temperature로 테스트
```bash
sweagent run \
  --config config/tamu_config_improved.yaml \
  --agent.model.temperature 0.0 \
  --agent.model.name "openai/protected.gpt-5" \
  --env.repo.github_url "https://github.com/django/django" \
  --problem_statement.github_url "https://github.com/django/django/issues/11099"
```

## 📊 LLM 출력 확인 방법

실행 중 다음과 같은 로그를 볼 수 있습니다:

```
🤖 INFO     📤 [LLM OUTPUT] ==== START ====
🤖 INFO     DISCUSSION
            I will first examine the validators file to understand...
            
            ```
            str_replace_editor view /django__django/django/contrib/auth/validators.py
            ```
🤖 INFO     📤 [LLM OUTPUT] ==== END ====
```

## 🎨 주요 개선 포인트

### Before (문제가 있던 방식)
```yaml
temperature: 0.7  # 너무 높음
system_template: "You are a helpful assistant..."  # 너무 간단
```

LLM 출력:
```
I'm going to investigate... [긴 설명]
```
→ Format error! (DISCUSSION 키워드 없음)

### After (개선된 방식)
```yaml
temperature: 0.0  # 안정적
system_template: |
  CRITICAL RULES:
  1. Always use EXACTLY ONE "DISCUSSION" section
  2. Keep bash commands SIMPLE
  ...
```

LLM 출력:
```
DISCUSSION
I'll check the file first.

```
str_replace_editor view /path/to/file
```
```
→ ✅ 성공!

## 🔧 추가 최적화 옵션

### Config 파일에서 직접 수정 가능:

```yaml
model:
  temperature: 0.0    # 0.0 ~ 0.2 추천
  top_p: 0.9         # 추가 가능
  
  # 특정 모델 시도
  # name: openai/protected.Claude Sonnet 4.5
  # name: openai/protected.o3
  # name: openai/protected.Claude Opus 4.1
```

## 📈 예상 개선 효과

| 지표 | 이전 | 개선 후 (예상) |
|------|------|----------------|
| Format Error | 높음 | 낮음 |
| Bash Syntax Error | 높음 | 중간 |
| 성공률 | ~30% | ~70-80% |
| 안정성 | 불안정 | 안정적 |

## 🐛 여전히 문제가 발생한다면?

1. **Temperature를 더 낮춰보기**
   ```bash
   --agent.model.temperature 0.0
   ```

2. **Claude 모델 시도**
   ```bash
   --agent.model.name "openai/protected.Claude Sonnet 4.5"
   ```

3. **더 간단한 이슈로 테스트**
   - 복잡한 bash 명령어가 필요 없는 이슈 선택

4. **로그 분석**
   - `📤 [LLM OUTPUT]` 부분을 확인하여 무엇이 잘못되었는지 파악

## 💡 Pro Tips

1. **Streaming 출력 실시간 확인**
   - 로그를 보면 LLM이 실시간으로 무엇을 생성하는지 볼 수 있음

2. **Format 패턴 학습**
   - 성공한 출력을 보고 패턴을 학습하면 프롬프트 개선 가능

3. **모델별 특성**
   - GPT-5: 창의적이지만 format 준수가 약함
   - Claude: Format 준수가 강하고 코드 생성 우수
   - O3: 추론 능력 강함

## 📝 로그 파일 위치

실행 후 전체 로그는 여기에 저장됩니다:
```
/Users/kylekim/Desktop/TAMU/CS689/course_project/SWE-agent/trajectories/
```

## ✅ 체크리스트

- [x] TAMU API 연동 완료
- [x] Streaming 처리 완료  
- [x] Temperature 최적화 완료
- [x] 프롬프트 개선 완료
- [x] LLM 출력 로깅 추가 완료
- [ ] 실제 테스트 및 검증 ← 다음 단계!

## 🎓 학습 포인트

이 프로젝트를 통해 배운 것:
1. LiteLLM을 통한 커스텀 API 연동
2. Streaming response 처리
3. Prompt engineering 기법
4. Temperature와 출력 안정성의 관계
5. Bash escaping과 heredoc의 복잡성
