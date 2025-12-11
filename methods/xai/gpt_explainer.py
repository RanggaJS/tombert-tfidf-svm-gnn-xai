import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# Default mapping for Twitter2015 sentiment labels
DEFAULT_CLASS_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def _format_probabilities(probabilities: Optional[List[float]], class_names: Dict[Any, str]) -> str:
    """Format probability list to a readable string."""
    if probabilities is None:
        return "Probabilitas tidak tersedia."

    # Pair with class names if provided
    pairs = []
    for idx, prob in enumerate(probabilities):
        label = class_names.get(idx, str(idx))
        pairs.append(f"{label}: {prob:.3f}")
    return "; ".join(pairs)


def generate_gpt_explanations(
    samples: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 320,
    class_names: Optional[Dict[Any, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate short XAI-style explanations for sentiment predictions using OpenAI GPT models.

    Args:
        samples: List of dict with keys:
            - text (str): Input text.
            - prediction (int/str): Predicted label id or name.
            - probabilities (optional list[float]): Class probabilities for the prediction order.
            - true_label (optional): Ground truth label id/name.
        model: OpenAI chat model name (default: gpt-4o-mini).
        temperature: Sampling temperature for the model.
        max_tokens: Maximum tokens for the explanation output.
        class_names: Optional mapping from label id to human-readable name.

    Returns:
        List of dict containing explanation and metadata for each sample.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY tidak ditemukan. Set env var OPENAI_API_KEY sebelum menggunakan XAI GPT.")

    client = OpenAI(api_key=api_key)
    class_names = class_names or DEFAULT_CLASS_NAMES

    results = []
    for idx, sample in enumerate(samples):
        text = sample.get("text", "")
        prediction = sample.get("prediction")
        probabilities = sample.get("probabilities")
        true_label = sample.get("true_label")
        image_caption = sample.get("image_caption") or sample.get("image_description")
        image_path = sample.get("image_path") or sample.get("image_id") or sample.get("image")

        # Resolve label names
        pred_name = class_names.get(prediction, str(prediction))
        true_name = class_names.get(true_label, str(true_label)) if true_label is not None else None

        prob_str = _format_probabilities(probabilities, class_names)

        system_prompt = (
            "Kamu adalah asisten XAI yang menjelaskan prediksi sentimen secara ringkas dan mudah dipahami. "
            "Jelaskan dalam 3-5 kalimat (bahasa Indonesia), fokus pada kata/frasa kunci dan konteks yang "
            "mendorong model memilih label tersebut."
        )

        user_prompt = (
            f"Teks: \"{text}\"\n"
            f"Label prediksi: {pred_name}\n"
            f"Probabilitas: {prob_str}\n"
        )
        # Include visual context if available
        if image_caption:
            user_prompt += f"Deskripsi gambar: {image_caption}\n"
        elif image_path:
            user_prompt += f"Referensi gambar (path/id): {image_path}\n"
        else:
            user_prompt += "Deskripsi gambar: (tidak tersedia)\n"
        if true_name:
            user_prompt += f"Label ground truth (jika ada): {true_name}\n"
        user_prompt += "Berikan alasan singkat mengapa model memilih label tersebut."

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            explanation = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Gagal membuat penjelasan GPT untuk sampel {idx}: {e}")
            explanation = f"[Gagal menghasilkan penjelasan: {e}]"

        results.append({
            "index": idx,
            "text": text,
            "prediction": prediction,
            "prediction_name": pred_name,
            "probabilities": probabilities,
            "true_label": true_label,
            "true_label_name": true_name,
            "image_caption": image_caption,
            "image_path": image_path,
            "explanation": explanation,
            "model": model,
        })

    return results

