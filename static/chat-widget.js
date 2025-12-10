(function () {
  // --- Inject Styles ---
  const style = document.createElement("style");
  style.innerHTML = `
        #chatbot-bubble {
            position: fixed;
            bottom: 25px;
            right: 25px;
            background: #0a8a83;
            color: white;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 9;
        }
        #chatbot-modal {
            display: none;
            position: fixed;
            right: 25px;
            bottom: 0;
            width: 350px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 0 20px rgba(0,0,0,0.2);
            flex-direction: column;
            z-index: 9;
        }
        #chatbot-header {
            background: #0a8a83;
            color: white;
            padding: 12px;
            font-size: 18px;
            border-radius: 12px 12px 0 0; 
        }
        #chatbot-close {
            float: right;
            cursor: pointer;
            font-size: 24px;
            align-self: center;
        }
        #chatbot-messages {
            height: 300px;
            overflow-y: auto;
            padding: 10px;
            background: #f7f7f7
        }
        #chatbot-input-area {
            display: flex;
            padding: 10px;
            gap: 5px;
        }
        .user-msg {
            background: #d1ffe0;
            padding: 8px;
            margin: 5px 0;
            border-radius: 8px;
        }
        .bot-msg {
            background: #e8e8ff;
            padding: 8px;
            margin: 5px 0;
            border-radius: 8px;
        }
        #chatbot-input {
            flex: 1;
            padding: 8px;
        }
    `;
    document.head.appendChild(style);

    // --- Bubble ---
    const bubble = document.createElement("div");
    bubble.id = "chatbot-bubble";
    bubble.innerHTML = "Chat";
    document.body.appendChild(bubble);

    // --- Modal ---
    const modal = document.createElement("div");
    modal.id = "chatbot-modal";
    modal.innerHTML = `
        <div id="chatbot-header">
            Ask me anything
            <span id="chatbot-close">&times;</span>
        </div>
        <div id="chatbot-messages"></div>
        <div id="chatbot-input-area">
            <input id="chatbot-input" type="text" placeholder="Type a question ..."/>
            <button id="chatbot-send">Send</button>
        </div>
    `;
    document.body.appendChild(modal);

    const messages = document.getElementById("chatbot-messages");
    const input = document.getElementById("chatbot-input");

    // Open & Close
    bubble.onclick = () =>
        (document.getElementById("chatbot-modal").style.display = "block");
    document.getElementById("chatbot-close").onclick = () => {
        modal.style.display = "none";
    };

      // Append Messages
    function addMessage(text, cls) {
        const div = document.createElement("div");
        div.className = cls;
        div.innerHTML = text;
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
    } 

    async function sendChat() {
        const question = input.value.trim();
        if (!question) return;

        addMessage(question, "user-msg");
        input.value = "";

        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        const data = await res.json();
        
        addMessage(
            `<b>${data.answer.replace("->", ": ")}</b><br><small><i>${
            data.reference[0]['path']
            }</i></small>`,
            "bot-msg"
        );
    }

    document.getElementById("chatbot-send").onclick = sendChat;
    input.addEventListener("keypress", (e) => e.key === "Enter" && sendChat());
})();
