import json
import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openai import OpenAI

from supply_chain_env.environment import Action, Observation, SupplyChainEnv
from supply_chain_env.inference import inference_agent


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_NAME = "supply-chain-env"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_action(action: List[float]) -> str:
    rounded = [round(float(x), 2) for x in action]
    return json.dumps(rounded, separators=(",", ":"))


def _safe_error(msg: Optional[str]) -> str:
    if not msg:
        return "null"
    return " ".join(str(msg).strip().split())


def _fallback_action(obs: Observation) -> List[float]:
    return [float(x) for x in inference_agent(obs).orders]


def _llm_action(client: Optional[OpenAI], obs: Observation) -> Tuple[List[float], Optional[str]]:
    if client is None:
        return _fallback_action(obs), "null"

    try:
        prompt = {
            "inventory": obs.inventory,
            "time": obs.time,
            "demand_history": obs.demand_history,
            "weather": obs.weather,
            "promotion": obs.promotion,
            "sustainability_score": obs.sustainability_score,
            "num_products": len(obs.inventory),
        }
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Return only a JSON array of non-negative numbers for order quantities.",
                },
                {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
            ],
            temperature=0,
            max_tokens=80,
        )
        content = (resp.choices[0].message.content or "").strip()
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise ValueError("model_output_not_json_array")
        parsed = json.loads(content[start : end + 1])
        if not isinstance(parsed, list):
            raise ValueError("model_output_not_list")
        out: List[float] = []
        n = len(obs.inventory)
        for i in range(n):
            val = 0.0
            if i < len(parsed):
                val = max(0.0, float(parsed[i]))
            out.append(val)
        return out, "null"
    except Exception as e:
        return _fallback_action(obs), str(e)


def _run_task(task: str, seed: int = 42) -> None:
    client: Optional[OpenAI] = None
    if HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}")

    rewards: List[float] = []
    success = True

    try:
        env = SupplyChainEnv(task=task, seed=seed)
        obs = env.reset(seed=seed)
        done = False
        step = 0

        while not done:
            action_vals, err = _llm_action(client, obs)
            action = Action(orders=action_vals)
            obs, reward, done, _info = env.step(action)
            rewards.append(float(reward))
            step += 1
            print(
                f"[STEP] step={step} action={_format_action(action.orders)} "
                f"reward={reward:.2f} done={_format_bool(done)} error={_safe_error(err)}"
            )

        reward_csv = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success=true steps={step} rewards={reward_csv}")
    except Exception as e:
        success = False
        step_count = len(rewards)
        print(
            f"[STEP] step={step_count + 1} action=[] reward=0.00 done=true error={_safe_error(str(e))}"
        )
        reward_csv = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={_format_bool(success)} steps={step_count} rewards={reward_csv}")


def main() -> None:
    for i, task in enumerate(("easy", "medium", "hard")):
        _run_task(task=task, seed=42 + i)


if __name__ == "__main__":
    main()
