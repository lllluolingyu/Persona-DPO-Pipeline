"""
Constitutional AI – DPO Dataset Generation Pipeline

Generates Direct Preference Optimization training pairs by running seed
prompts through: Student Generation → Constitutional Critique → Revision.
"""

import asyncio
import json
import logging
import os
import random
import time
from glob import glob
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dpo_pipeline")


# ── Configuration ─────────────────────────────────────────────────────────


def load_config() -> dict:
    """Load all tunables from the .env file."""
    load_dotenv()
    return {
        "student_base_url": os.getenv("STUDENT_API_BASE_URL"),
        "student_api_key": os.getenv("STUDENT_API_KEY"),
        "student_model": os.getenv("STUDENT_MODEL"),
        "teacher_base_url": os.getenv("TEACHER_API_BASE_URL"),
        "teacher_api_key": os.getenv("TEACHER_API_KEY"),
        "teacher_model": os.getenv("TEACHER_MODEL"),
        "reviewer_model": os.getenv("REVIEWER_MODEL"),
        "enable_critique": os.getenv("ENABLE_CRITIQUE_REVISION", "true").lower()
        == "true",
        "temperature": float(os.getenv("TEMPERATURE", "1.0")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "16384")),
        "max_concurrent": int(os.getenv("MAX_CONCURRENT_REQUESTS", "20")),
        "max_retries": int(os.getenv("MAX_RETRIES", "10")),
        "retry_base_delay": float(os.getenv("RETRY_BASE_DELAY", "2.0")),
    }


# ── Data Ingestion ────────────────────────────────────────────────────────


def load_prompts(data_dir: str = "data") -> list[dict]:
    """Load prompts from every .jsonl file in *data_dir*.

    Two formats are recognised:
      • ``{"prompt": "..."}``
      • ``{"conversations": [{"role": "user", "content": "..."}, ...]}``

    Returns a flat list of ``{"prompt": str, "messages": list|None, "source": str}``.
    """
    items: list[dict] = []
    for path in sorted(glob(os.path.join(data_dir, "*.jsonl"))):
        fname = os.path.basename(path)
        count = 0
        with open(path, encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("Invalid JSON at %s:%d – skipping", fname, line_no)
                    continue

                if "prompt" in obj:
                    items.append(
                        {"prompt": obj["prompt"], "messages": None, "source": fname}
                    )
                elif "conversations" in obj:
                    convs = obj["conversations"]
                    if not convs:
                        continue
                    # Find last user turn
                    last_user_idx = next(
                        (
                            i
                            for i in range(len(convs) - 1, -1, -1)
                            if convs[i]["role"] == "user"
                        ),
                        None,
                    )
                    if last_user_idx is None:
                        continue
                    prompt_text = convs[last_user_idx]["content"]

                    # Build context: all turns up to (excluding) the last assistant reply
                    context = []
                    for msg in convs:
                        if msg["role"] in ("user", "assistant"):
                            # Skip the final assistant turn – that's what we regenerate
                            if msg["role"] == "assistant" and msg is convs[-1]:
                                continue
                            context.append(
                                {"role": msg["role"], "content": msg["content"]}
                            )

                    items.append(
                        {
                            "prompt": prompt_text,
                            "messages": context if len(context) > 1 else None,
                            "source": fname,
                        }
                    )
                else:
                    log.warning(
                        "Unrecognised format at %s:%d – skipping", fname, line_no
                    )
                count += 1
        log.info("Loaded %d items from %s", count, fname)
    log.info("Total prompts: %d", len(items))
    return items


# ── LLM helper ────────────────────────────────────────────────────────────


async def call_llm(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 10,
    retry_base_delay: float = 2.0,
) -> str:
    """Send a chat-completion request with retry + exponential back-off."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt) + random.uniform(0, 1)
                log.warning(
                    "LLM call failed (attempt %d/%d): %s – retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                log.error("LLM call failed after %d attempts: %s", max_retries, e)
                raise


# ── Pipeline Steps ────────────────────────────────────────────────────────

CRITIQUE_TEMPLATE = """\
请你作为一位严格的宪法审查员，根据提供的《行为准则》对以下回复进行评判。

## 用户原始提问
{prompt}

## 待评审的回复
{response}

## 你的任务
请根据《行为准则》中的各项维度（如 [PERSONA-CONSISTENCY]、[HONESTY]、\
[HELPFULNESS-DEPTH]、[SAFETY-ETHICS] 等），对上述回复进行逐条审查。

对于每一个发现的问题，请按以下格式输出：

[维度标签] 简述判断理由。
严重程度：严重偏差 / 风格偏好

如果回复整体表现良好，也请指出值得保持的优点。

最后，请给出一段总结性的修改建议，说明应当如何改进这份回复以更好地符合宪法。"""

REVISION_TEMPLATE = """\
请根据以下信息，生成一份修订后的回复。修订后的回复应当解决审查意见中\
指出的所有问题，同时严格遵守你的行为准则。

## 用户原始提问
{prompt}

## 原始回复
{response}

## 审查意见
{critique}

## 你的任务
请直接输出修订后的回复内容。不要包含任何解释、元评论或对审查意见的引用\
——只输出最终的、面向用户的回复。"""


class StudentGenerator:
    """Generates initial baseline responses using the student model."""

    def __init__(
        self, client: AsyncOpenAI, model: str, constitution: str, cfg: dict
    ):
        self.client = client
        self.model = model
        self.constitution = constitution
        self.cfg = cfg

    async def generate(
        self,
        prompt: str,
        semaphore: asyncio.Semaphore,
        messages: list[dict] | None = None,
    ) -> str:
        msgs = [{"role": "system", "content": self.constitution}]
        if messages:
            msgs.extend(messages)
        else:
            msgs.append({"role": "user", "content": prompt})
        return await call_llm(
            self.client,
            self.model,
            msgs,
            temperature=self.cfg["temperature"],
            max_tokens=self.cfg["max_tokens"],
            semaphore=semaphore,
            max_retries=self.cfg["max_retries"],
            retry_base_delay=self.cfg["retry_base_delay"],
        )


class ConstitutionalCritique:
    """Generates a critique of a response against the constitution."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        reviewer_constitution: str,
        cfg: dict,
    ):
        self.client = client
        self.model = model
        self.reviewer_constitution = reviewer_constitution
        self.cfg = cfg

    async def critique(
        self, prompt: str, response: str, semaphore: asyncio.Semaphore
    ) -> str:
        msgs = [
            {"role": "system", "content": self.reviewer_constitution},
            {
                "role": "user",
                "content": CRITIQUE_TEMPLATE.format(
                    prompt=prompt, response=response
                ),
            },
        ]
        return await call_llm(
            self.client,
            self.model,
            msgs,
            temperature=self.cfg["temperature"],
            max_tokens=self.cfg["max_tokens"],
            semaphore=semaphore,
            max_retries=self.cfg["max_retries"],
            retry_base_delay=self.cfg["retry_base_delay"],
        )


class ConstitutionalReviser:
    """Generates a revised response that addresses the critique."""

    def __init__(
        self, client: AsyncOpenAI, model: str, constitution: str, cfg: dict
    ):
        self.client = client
        self.model = model
        self.constitution = constitution
        self.cfg = cfg

    async def revise(
        self,
        prompt: str,
        response: str,
        critique: str,
        semaphore: asyncio.Semaphore,
    ) -> str:
        msgs = [
            {"role": "system", "content": self.constitution},
            {
                "role": "user",
                "content": REVISION_TEMPLATE.format(
                    prompt=prompt, response=response, critique=critique
                ),
            },
        ]
        return await call_llm(
            self.client,
            self.model,
            msgs,
            temperature=self.cfg["temperature"],
            max_tokens=self.cfg["max_tokens"],
            semaphore=semaphore,
            max_retries=self.cfg["max_retries"],
            retry_base_delay=self.cfg["retry_base_delay"],
        )


# ── Pipeline Orchestration ────────────────────────────────────────────────


class DPOPipeline:
    """Runs the full Constitutional-AI → DPO-pair generation pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.semaphore = asyncio.Semaphore(config["max_concurrent"])

        self.constitution = Path("constitution.md").read_text(encoding="utf-8")
        self.reviewer_constitution = Path("constitution_reviewer.md").read_text(
            encoding="utf-8"
        )

        student_client = AsyncOpenAI(
            base_url=config["student_base_url"],
            api_key=config["student_api_key"],
        )
        teacher_client = AsyncOpenAI(
            base_url=config["teacher_base_url"],
            api_key=config["teacher_api_key"],
        )

        self.student = StudentGenerator(
            student_client, config["student_model"], self.constitution, config
        )
        self.critiquer = ConstitutionalCritique(
            teacher_client,
            config["reviewer_model"],
            self.reviewer_constitution,
            config,
        )
        self.reviser = ConstitutionalReviser(
            teacher_client, config["teacher_model"], self.constitution, config
        )
        self.teacher_client = teacher_client
        self.enable_critique = config["enable_critique"]

        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    # ── single-item processing ────────────────────────────────────────

    async def _process_item(
        self, item: dict, idx: int, total: int
    ) -> dict | None:
        prompt = item["prompt"]
        messages = item.get("messages")

        try:
            # 1) Student generates initial response
            initial = await self.student.generate(
                prompt, self.semaphore, messages
            )

            if self.enable_critique:
                # 2) Critique
                critique = await self.critiquer.critique(
                    prompt, initial, self.semaphore
                )
                # 3) Revision
                revised = await self.reviser.revise(
                    prompt, initial, critique, self.semaphore
                )
            else:
                # Single-pass: teacher generates the "chosen" response directly
                critique = None
                msgs = [{"role": "system", "content": self.constitution}]
                if messages:
                    msgs.extend(messages)
                else:
                    msgs.append({"role": "user", "content": prompt})
                revised = await call_llm(
                    self.teacher_client,
                    self.config["teacher_model"],
                    msgs,
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                    semaphore=self.semaphore,
                    max_retries=self.config["max_retries"],
                    retry_base_delay=self.config["retry_base_delay"],
                )

            log.info("Processed %d/%d [%s]", idx + 1, total, item.get("source", ""))

            dpo_pair: dict = {
                "prompt": prompt,
                "chosen": revised,
                "rejected": initial,
            }
            if messages:
                dpo_pair["messages"] = messages

            metadata = {
                "source": item.get("source", "unknown"),
                "critique": critique,
            }
            return {"dpo": dpo_pair, "metadata": metadata}

        except Exception as e:
            log.error(
                "Failed item %d (%s…): %s", idx + 1, prompt[:50], e
            )
            return None

    # ── batch orchestration ───────────────────────────────────────────

    async def run(self, items: list[dict]):
        total = len(items)
        log.info(
            "Starting pipeline: %d items, concurrency=%d, critique=%s",
            total,
            self.config["max_concurrent"],
            self.enable_critique,
        )

        ts = time.strftime("%Y%m%d_%H%M%S")
        dpo_path = self.output_dir / f"dpo_pairs_{ts}.jsonl"
        meta_path = self.output_dir / f"dpo_metadata_{ts}.jsonl"

        write_lock = asyncio.Lock()
        stats = {"completed": 0, "failed": 0}

        async def process_and_save(item: dict, idx: int):
            result = await self._process_item(item, idx, total)
            if result:
                async with write_lock:
                    with open(dpo_path, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(result["dpo"], ensure_ascii=False) + "\n"
                        )
                    with open(meta_path, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(result["metadata"], ensure_ascii=False)
                            + "\n"
                        )
                    stats["completed"] += 1
            else:
                stats["failed"] += 1

        tasks = [
            asyncio.create_task(process_and_save(item, i))
            for i, item in enumerate(items)
        ]
        await asyncio.gather(*tasks)

        log.info(
            "Done. %d succeeded, %d failed. Output → %s",
            stats["completed"],
            stats["failed"],
            dpo_path,
        )


# ── Entry point ───────────────────────────────────────────────────────────


async def main():
    config = load_config()
    items = load_prompts("data")
    if not items:
        log.error("No prompts found in data/. Exiting.")
        return
    pipeline = DPOPipeline(config)
    await pipeline.run(items)


if __name__ == "__main__":
    asyncio.run(main())
