# Архив экспериментов (Archive of Previous Attempts)

В этой папке сохранены экспериментальные скрипты из прошлых попыток решения задачи, которые исследовались в процессе соревнования:
- **`finetune_article_minilm_fold0.py` и `final_article_submission.py`** — эксперименты с открытой моделью `cross-encoder/mmarco-mMiniLMv2` (показали просадку на паблике из-за вытеснения правильных сопутствующих статей чужими кандидатами).
- **`jina_gguf_pilot.py` и `jina_gguf_full.py`** — эксперименты с открытым cross-encoder `Jina-reranker-v3-gguf`.
- **`exp_set_decoder.py`, `exp_set_gating.py` и др.** — эксперименты с прямым сетовым декодированием и альтернативными гейтами.

---

### 🟢 Финальное рабочее решение находится в корневом каталоге проекта:
- `write_final_submission.py` — главный генератор финального ответа `answer.csv`.
- `exp_intent_closed_set.py` — классификатор закрытого множества интентов (Multi-label SVM).
- `exp_ensemble_audit.py` — 5-фолдовый аудит и проверка гипотезы `overlap3_not_1`.
- `ltr_solution.py` и `solution.py` — базовый поиск и предобработка текста.
