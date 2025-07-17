# 檔案位置: backend/ai_pipeline.py

import torch
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import jiwer
import soundfile as sf
import numpy as np
from tqdm import tqdm
import jieba
from opencc import OpenCC
# -------------------------------------------------------------------
# 導入所有需要的函式庫
# -------------------------------------------------------------------

# 翻譯模型 (NLLB)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 語音辨識模型 (Whisper)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 語音合成模型 (MaskGCT from Amphion)
# 由於 models, third_party, utils 資料夾現在位於 backend 內部，
# Python 可以直接找到並導入它們，無需任何路徑操作。
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
        self.cc = OpenCC('t2s')
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
        # 使用本地相對路徑來讀取設定檔
        cfg_path = "./models/tts/maskgct/config/maskgct.json"
        cfg = load_config(cfg_path)
        
        # 💡 關鍵修改處：將 semantic_model 的設定傳入 build_semantic_model 函式
        # 這會讓它從本地 ckpt 載入，而不是從 transformers 下載
        # semantic_model, semantic_mean, semantic_std = build_semantic_model(
        #     cfg.model.semantic_model, self.device
        # )
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
        s2a_1layer_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors")
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
        
        # 關鍵修改：動態識別 prompt 音訊的內容，使其更穩健
        print(f"    正在識別音色樣本 '{os.path.basename(prompt_wav_path)}' 的內容...")
        prompt_lang = "en" # 假設所有音色樣本都是英文
        prompt_text = self.recognize(prompt_wav_path, prompt_lang)
        print(f"    識別出的音色樣本內容: '{prompt_text[:30]}...'")
        
        with torch.cuda.amp.autocast():
            recovered_audio = self.maskgct_pipeline.maskgct_inference(
                prompt_wav_path, prompt_text, text, prompt_lang, lang, target_len=None
            )
        return recovered_audio, 24000

    def recognize(self, audio_path: str, lang: str) -> str:
            """
            使用 Whisper 模型识别语音。
            (最终版：在输出中文结果后，立即统一为简体)
            """
            print(f"执行语音识别: {audio_path} (语言: {lang})")
            
            # 调用 Whisper 模型
            result = self.whisper_pipe(
                audio_path,
                generate_kwargs={"language": lang, "task": "transcribe"}
            )
            recognized_text = result.get("text", "").strip()

            # 核心逻辑：如果是中文，立即进行简繁统一
            if lang == 'chinese':
                print("      检测到中文识别结果，执行简繁统一...")
                try:
                    simplified_text = self.cc.convert(recognized_text)
                    return simplified_text
                except Exception as e:
                    print(f"      [警告] OpenCC 简繁转换失败: {e}")
                    # 即使转换失败，也返回原始识别结果
                    return recognized_text
            
            # 如果不是中文，直接返回原始识别结果
            return recognized_text
    


    # 在 backend/ai_pipeline.py 文件中，找到并替换这个函数

    def evaluate(self, reference_text: str, hypothesis_text: str, lang_code: str) -> dict:
        """
        使用 Jiwer 计算 WER 和 CER。
        (最终版：采用标准范式处理中文分词，以解决 jiwer 的解析错误)
        """
        print(f"执行评估 (语言: {lang_code})...")
        
        # 1. 定义通用的文本标准化规则
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
        ])
        
        # 2. 对两个字符串应用标准化规则
        processed_reference = transformation(reference_text)
        processed_hypothesis = transformation(hypothesis_text)

        # 3. 智能计算 WER (最终修正版)
        if lang_code == 'zh':
            print("      检测到中文，使用 jieba 分词并用空格连接...")
            # 对于中文，先用 jieba 分词，然后用空格将词语连接成一个标准字符串
            ref_words = " ".join(jieba.lcut(processed_reference))
            hyp_words = " ".join(jieba.lcut(processed_hypothesis))
            
            # 将两个处理好的、空格分隔的字符串交给 jiwer
            wer_score = jiwer.wer(ref_words, hyp_words)
        else:
            # 对于其他语言，直接使用 jiwer 默认的空格分词
            wer_score = jiwer.wer(processed_reference, processed_hypothesis)
            
        # 4. CER 是基于字符的，不受分词影响，可以直接计算
        # 我们仍然使用处理过的字符串，以确保公平性 (移除了标点和多余空格)
        cer_score = jiwer.cer(processed_reference, processed_hypothesis)

        return {
            "wer": wer_score,
            "cer": cer_score,
        }
# --- 全局實例 ---
# 在應用程式啟動時，只建立一次 AIPipeline 的實例。
# FastAPI 主程式 (main.py) 將會導入並使用這個物件。
ai_pipeline = AIPipeline()
