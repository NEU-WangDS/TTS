# backend/ai_pipeline.py

import torch
import os
import jiwer
import soundfile as sf
import numpy as np
from tqdm import tqdm

# -------------------------------------------------------------------
# 導入所有需要的函式庫
# -------------------------------------------------------------------

# 翻譯模型 (NLLB)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 語音辨識模型 (Whisper)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 語音合成模型 (MaskGCT from Amphion)
# 注意：這裡假設您的 Amphion 專案路徑已經被正確地加入到 Python 的搜尋路徑中
# 您可能需要調整 sys.path 或將 Amphion 的 models 目錄複製到 backend 中
import sys
# 假設 Amphion 專案與您的 AI_Voice_App 在同一級目錄
# 如果不是，請修改此路徑
amphion_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Amphion'))
if amphion_path not in sys.path:
    sys.path.append(amphion_path)
    
from models.tts.maskgct.maskgct_utils import (
    build_semantic_model,
    build_semantic_codec,
    build_acoustic_codec,
    build_t2s_model,
    build_s2a_model,
    load_config,
    MaskGCT_Inference_Pipeline,
)
from huggingface_hub import hf_hub_download
import safetensors


class AIPipeline:
    """
    一個封裝了所有 AI 模型（翻譯、TTS、ASR）的管線類別。
    模型只在伺服器啟動時載入一次。
    """
    def __init__(self):
        print("="*50)
        print("正在初始化 AI 管線，開始載入所有模型...")
        print("="*50)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # --- 1. 載入 NLLB 翻譯模型 ---
        self._load_translator_model()

        # --- 2. 載入 MaskGCT TTS 模型 ---
        self._load_tts_model()

        # --- 3. 載入 Whisper ASR 模型 ---
        self._load_asr_model()
        
        print("\n✅ 所有 AI 模型準備就緒！伺服器可以開始接收請求。")

    def _load_translator_model(self):
        print("\n[1/3] 正在載入 NLLB 翻譯模型...")
        model_id = "facebook/nllb-200-distilled-600M"
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        self.translator_tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("NLLB 翻譯模型載入完成。")

    def _load_tts_model(self):
        print("\n[2/3] 正在載入 MaskGCT TTS 模型...")
        # 這裡的程式碼來自您之前的 Amphion 腳本
        cfg_path = os.path.join(amphion_path, "models/tts/maskgct/config/maskgct.json")
        cfg = load_config(cfg_path)
        
        semantic_model, semantic_mean, semantic_std = build_semantic_model(self.device)
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, self.device)
        codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, self.device)
        t2s_model = build_t2s_model(cfg.model.t2s_model, self.device)
        s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, self.device)
        s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, self.device)

        print("    正在從 Hugging Face 下載 MaskGCT 模型權重...")
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        codec_encoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model.safetensors")
        codec_decoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors")
        t2s_model_ckpt = hf_hub_download("amphion/MaskGCT", filename="t2s_model/model.safetensors")
        s2a_1layer_ckpt = hf_hub_download("amphion/MaskGT", filename="s2a_model/s2a_model_1layer/model.safetensors")
        s2a_full_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors")

        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
        safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
        safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
        safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
        safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)
        
        self.maskgct_pipeline = MaskGCT_Inference_Pipeline(
            semantic_model, semantic_codec, codec_encoder, codec_decoder, t2s_model,
            s2a_model_1layer, s2a_model_full, semantic_mean, semantic_std, self.device
        )
        print("MaskGCT TTS 模型載入完成。")

    def _load_asr_model(self):
        print("\n[3/3] 正在載入 Whisper ASR 模型...")
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)
        
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("Whisper ASR 模型載入完成。")

    def translate(self, text: str, target_lang_code: str, src_lang_code: str = "eng_Latn") -> str:
        """
        使用 NLLB 模型翻譯文本。
        """
        print(f"執行翻譯: {text[:20]}... -> {target_lang_code}")
        self.translator_tokenizer.src_lang = src_lang_code
        inputs = self.translator_tokenizer(text, return_tensors="pt").to(self.device)
        translated_tokens = self.translator_model.generate(
            **inputs, forced_bos_token_id=self.translator_tokenizer.lang_code_to_id[target_lang_code]
        )
        translated_text = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text

    def synthesize(self, text: str, lang: str, prompt_wav_path: str) -> (np.ndarray, int):
        """
        使用 MaskGCT 模型合成語音。
        返回 (音訊 NumPy 陣列, 採樣率)。
        """
        print(f"執行語音合成: {text[:20]}... (語言: {lang})")
        # 假設 prompt 永遠是英文
        prompt_text = "This is the prompt to provide voice timbre."
        prompt_lang = "en"
        
        with torch.cuda.amp.autocast():
            recovered_audio = self.maskgct_pipeline.maskgct_inference(
                prompt_wav_path, prompt_text, text, prompt_lang, lang, target_len=None
            )
        return recovered_audio, 24000

    def recognize(self, audio_path: str, lang: str) -> str:
        """
        使用 Whisper 模型識別語音。
        """
        print(f"執行語音辨識: {audio_path} (語言: {lang})")
        result = self.whisper_pipe(
            audio_path,
            generate_kwargs={"language": lang, "task": "transcribe"}
        )
        return result["text"]
    
    def evaluate(self, reference_text: str, hypothesis_text: str) -> dict:
        """
        使用 Jiwer 計算 WER 和 CER。
        """
        print("執行評估...")
        measures = jiwer.compute_measures(reference_text, hypothesis_text)
        return {
            "wer": measures.get('wer', 1.0),
            "cer": measures.get('cer', 1.0),
        }

# --- 全局實例 ---
# 在應用程式啟動時，只建立一次 AIPipeline 的實例。
# FastAPI 主程式 (main.py) 將會導入並使用這個物件。
ai_pipeline = AIPipeline()
