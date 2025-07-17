# 檔案位置: backend/main.py

import os
import sys

# 💡 =================================================================================
# 💡 關鍵修改：設定 Hugging Face 快取路徑和鏡像站
# 💡 這段程式碼必須在導入任何 transformers 或 huggingface_hub 相關模組之前執行！
# 💡 =================================================================================

# --- 設定快取路徑到資料盤 ---
MODELS_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_and_data'))
HF_CACHE_DIR = os.path.join(MODELS_DATA_DIR, 'huggingface_cache')
os.makedirs(HF_CACHE_DIR, exist_ok=True)

print(f"Hugging Face 快取路徑已設定為: {HF_CACHE_DIR}")
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(HF_CACHE_DIR, 'hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_CACHE_DIR, 'hub')

# --- 💡 新增此行：設定 Hugging Face 官方鏡像站以解決網路問題 ---
print("正在設定 Hugging Face 鏡像站以加速下載...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 現在可以安全地導入我們的 AI 管線了，它內部會導入 transformers
from fastapi import FastAPI, HTTPException
# 💡 主要修改處：導入 FileResponse 用於返回 HTML 檔案
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import soundfile as sf
import uuid
import logging

# 導入我們在 ai_pipeline.py 中封裝好的 AI 管線物件
# 伺服器啟動時，ai_pipeline.py 會被執行，所有模型會被載入
try:
    # 前面的點 . 代表「從當前所在的資料夾(也就是 backend)導入」
    from ai_pipeline import ai_pipeline 
except ImportError as e:
    print("="*80)
    print("錯誤：無法導入 ai_pipeline。")
    print("請確保您已遵循專案結構，並且 ai_pipeline.py 能夠正確找到 Amphion 的路徑。")
    print(f"詳細錯誤: {e}")
    print("="*80)
    exit()

# --- 1. 初始化 FastAPI 應用 ---
app = FastAPI(
    title="多語言 AI 語音管線 API",
    description="一個整合了翻譯、TTS、ASR 和評估的 API。",
    version="1.0.0",
)

# --- 2. 定義資料模型 (用於驗證 API 請求的內容) ---
class ProcessRequest(BaseModel):
    text: str
    target_language: str
    prompt_wav_path: str

# --- 3. 靜態檔案與路徑設定 ---
# 這個路徑是相對於 main.py 所在的 backend/ 資料夾
STATIC_FILES_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static')
PROMPTS_DIR = os.path.join(MODELS_DATA_DIR, 'prompts')
AUDIO_OUTPUT_DIR = os.path.join(MODELS_DATA_DIR, 'generated_audio')

# 建立必要的資料夾
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# 將前端的 static 資料夾掛載到 FastAPI，這樣瀏覽器才能訪問到 index.html 等檔案
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")
# 將生成的音訊資料夾也掛載，以便前端可以播放音訊
app.mount("/generated_audio", StaticFiles(directory=AUDIO_OUTPUT_DIR), name="generated_audio")
# 將 prompts 資料夾也掛載，以便前端可以預覽音色（如果需要）
app.mount("/prompts", StaticFiles(directory=PROMPTS_DIR), name="prompts")


# --- 4. 語言代碼映射 ---
# 將前端傳來的簡單代碼，轉換為各個模型需要的特定代碼
NLLB_LANG_MAP = {
    "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "de": "deu_Latn", "fr": "fra_Latn"
}
WHISPER_LANG_MAP = {
    "zh": "chinese", "ja": "japanese", "ko": "korean",
    "de": "german", "fr": "french"
}

# --- 5. API 端點 (Endpoints) ---

# 💡 主要修改處：修改根目錄端點，讓它直接返回 index.html
@app.get("/", response_class=FileResponse)
def read_root():
    """根目錄，直接提供前端網頁"""
    return os.path.join(STATIC_FILES_DIR, 'index.html')

@app.get("/prompts", response_model=list[str])
def get_prompts():
    """獲取所有可用的音色樣本列表"""
    if not os.path.isdir(PROMPTS_DIR):
        return []
    # 返回可供前端直接使用的相對 URL 路徑
    return sorted([f"/prompts/{f}" for f in os.listdir(PROMPTS_DIR) if f.endswith('.wav')])


@app.post("/process")
async def process_full_pipeline(request: ProcessRequest):
    """
    接收前端請求，執行完整的 翻譯->TTS->ASR->評估 流程
    """
    print(f"\n收到新請求: 語言={request.target_language}, 音色='{request.prompt_wav_path}'")
    try:
        # --- 步驟 1: 翻譯 ---
        nllb_lang = NLLB_LANG_MAP.get(request.target_language)
        if not nllb_lang:
            raise HTTPException(status_code=400, detail=f"不支援的目標語言: {request.target_language}")
        
        translated_text = ai_pipeline.translate(request.text, nllb_lang)

        # --- 步驟 2: 語音合成 (TTS) ---
        # 將前端傳來的 URL 路徑轉換為伺服器上的絕對路徑
        prompt_absolute_path = os.path.join(os.path.dirname(__file__), '..', 'models_and_data', request.prompt_wav_path.lstrip('/'))
        if not os.path.exists(prompt_absolute_path):
             raise HTTPException(status_code=404, detail=f"找不到音色檔案: {prompt_absolute_path}")

        audio_data, samplerate = ai_pipeline.synthesize(
            translated_text, request.target_language, prompt_absolute_path
        )
        
        # 儲存生成的音訊檔案
        filename = f"{uuid.uuid4()}.wav"
        save_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
        sf.write(save_path, audio_data, samplerate)
        # 生成可供前端訪問的 URL
        audio_url = f"/generated_audio/{filename}"

        # --- 步驟 3: 語音辨識 (ASR) ---
        whisper_lang = WHISPER_LANG_MAP.get(request.target_language)
        transcribed_text = ai_pipeline.recognize(save_path, whisper_lang)

        # --- 步驟 4: 評估 (WER/CER) ---
        #evaluation_results = ai_pipeline.evaluate(translated_text, transcribed_text)
        # 将 request.target_language 作为新参数传递进去
        evaluation_results = ai_pipeline.evaluate(translated_text, transcribed_text, request.target_language)

        # --- 步驟 5: 返回完整結果 ---
        return {
            "original_text": request.text,
            "translated_text": translated_text,
            "transcribed_text": transcribed_text,
            "audio_url": audio_url,
            "wer": evaluation_results['wer'],
            "cer": evaluation_results['cer']
        }

    except Exception as e:
        logging.exception("處理請求時發生錯誤")
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {e}")
