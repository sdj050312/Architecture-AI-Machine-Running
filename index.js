const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function addMessage(text, isUser) {
    const msgDiv = document.createElement('div');
    msgDiv.className = isUser ? 'user-msg' : 'bot-msg';
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function getBotResponse(question) {
    // 여기에 나중에 진짜 AI(API)를 연결하면 됩니다.
    // 지금은 간단한 조건문으로 시뮬레이션해볼게요.
    if (question.includes("주소") || question.includes("동")) {
        return `🔍 분석 결과: 해당 부지는 '제2종 일반주거지역'으로 확인됩니다. 예상 건폐율은 60%이며, 주변 시공 단가는 평당 800만 원 선입니다.`;
    } else if (question.includes("비용")) {
        return `💰 예상 공사비: 단독주택 기준, 기초 공사 포함 약 3억 원 내외의 예산이 필요할 것으로 보입니다.`;
    } else {
        return `건축 공학적 관점에서 분석 중입니다... 질문하신 "${question}"에 대해 더 자세한 정보를 수집 중입니다.`;
    }
}

sendBtn.addEventListener('click', () => {
    const text = userInput.value;
    if (!text) return;

    addMessage(text, true); // 유저 메시지 추가
    userInput.value = '';

    setTimeout(() => {
        const response = getBotResponse(text);
        addMessage(response, false); // 봇 메시지 추가
    }, 600);
});

// 엔터키 지원
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendBtn.click();
});