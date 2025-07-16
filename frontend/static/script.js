// 档案位置: frontend/static/script.js
document.addEventListener('DOMContentLoaded', () => {
    // 获取所有需要操作的 DOM 元素
    const textInput = document.getElementById('text-input');
    const langSelect = document.getElementById('lang-select');
    const promptSelect = document.getElementById('prompt-select');
    const processButton = document.getElementById('process-button');
    const loadingDiv = document.getElementById('loading');
    const resultsCard = document.getElementById('results-card');
    const errorAlert = document.getElementById('error-alert');
    
    const audioResultDiv = document.getElementById('audio-result');
    const werResultSpan = document.getElementById('wer-result');
    const cerResultSpan = document.getElementById('cer-result');
    const originalTextSpan = document.getElementById('original-text');
    const translatedTextSpan = document.getElementById('translated-text');
    const transcribedTextSpan = document.getElementById('transcribed-text');

    // 动态从后端加载音色选项
    function loadPrompts() {
        fetch('/prompts')
            .then(response => {
                if (!response.ok) throw new Error('无法获取音色列表');
                return response.json();
            })
            .then(prompts => {
                promptSelect.innerHTML = ''; // 清空旧选项
                if (prompts.length === 0) {
                     const option = document.createElement('option');
                     option.textContent = '在 prompts 文件夹中找不到音色档案';
                     promptSelect.appendChild(option);
                     promptSelect.disabled = true;
                } else {
                    prompts.forEach(prompt => {
                        const option = document.createElement('option');
                        // 从 URL 中提取档名，例如 "/prompts/my_voice.wav" -> "my_voice.wav"
                        option.value = prompt; // value 仍然是完整的 URL 路径
                        option.textContent = prompt.split('/').pop();
                        promptSelect.appendChild(option);
                    });
                    promptSelect.disabled = false;
                }
            })
            .catch(error => {
                console.error('获取音色错误:', error);
                promptSelect.disabled = true;
                promptSelect.innerHTML = '<option>加载音色失败</option>';
            });
    }

    // "开始处理" 按钮的点击事件
    processButton.addEventListener('click', () => {
        const text = textInput.value;
        const language = langSelect.value;
        const promptWavPath = promptSelect.value;

        if (!text.trim() || !promptWavPath || promptSelect.disabled) {
            alert('请输入文本并选择一个有效的音色！');
            return;
        }

        // --- 更新 UI 状态，进入载入模式 ---
        loadingDiv.style.display = 'block';
        resultsCard.style.display = 'none';
        errorAlert.style.display = 'none';
        processButton.disabled = true;
        processButton.textContent = '处理中...';

        // --- 发送 API 请求到后端 ---
        fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, target_language: language, prompt_wav_path: promptWavPath })
        })
        .then(response => {
            if (!response.ok) {
                // 如果后端返回错误，尝试解析 JSON 错误讯息
                return response.json().then(err => { throw new Error(err.detail || '服务器返回了未知错误') });
            }
            return response.json();
        })
        .then(data => {
            // --- 成功收到结果，更新 UI ---
            resultsCard.style.display = 'block';

            // 填充结果数据
            originalTextSpan.textContent = data.original_text;
            translatedTextSpan.textContent = data.translated_text;
            transcribedTextSpan.textContent = data.transcribed_text;
            werResultSpan.textContent = `${(data.wer * 100).toFixed(2)}%`;
            cerResultSpan.textContent = `${(data.cer * 100).toFixed(2)}%`;
            
            // 创建并显示音讯播放器
            audioResultDiv.innerHTML = '';
            const audio = new Audio(data.audio_url);
            audio.controls = true;
            audio.className = 'w-100';
            audioResultDiv.appendChild(audio);
        })
        .catch(error => {
            // --- 处理过程中发生错误，显示错误提示 ---
            errorAlert.textContent = '处理失败：' + error.message;
            errorAlert.style.display = 'block';
            console.error('Error:', error);
        })
        .finally(() => {
            // --- 无论成功或失败，都恢复 UI 状态 ---
            loadingDiv.style.display = 'none';
            processButton.disabled = false;
            processButton.textContent = '开始处理';
        });
    });

    // 初始化页面，加载音色选项
    loadPrompts();
});
