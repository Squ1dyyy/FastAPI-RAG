"""Single-page HTML/JS interface for the FastAPI Hybrid RAG Chat with sessions."""

CHAT_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FastAPI Hybrid RAG Chat</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
	font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
	background: rgb(15, 23, 42);
	color: rgb(226, 232, 240);
	height: 100vh;
	display: flex;
	overflow: hidden;
}
.sidebar {
	width: 280px;
	background: rgb(30, 41, 59);
	border-right: 1px solid rgb(51, 65, 85);
	display: flex;
	flex-direction: column;
	padding: 16px;
}
.brand {
	font-size: 18px;
	font-weight: 700;
	color: rgb(56, 189, 248);
	margin-bottom: 12px;
}
.new-chat-btn {
	background: rgb(14, 165, 233);
	color: white;
	border: none;
	padding: 10px 14px;
	border-radius: 8px;
	font-weight: 600;
	cursor: pointer;
	margin-bottom: 16px;
	transition: opacity 0.15s;
}
.new-chat-btn:hover { opacity: 0.9; }
.sessions-list {
	flex: 1;
	overflow-y: auto;
	display: flex;
	flex-direction: column;
	gap: 6px;
}
.session-item {
	padding: 10px 12px;
	border-radius: 8px;
	background: rgb(15, 23, 42);
	cursor: pointer;
	display: flex;
	justify-content: space-between;
	align-items: center;
	font-size: 14px;
	border: 1px solid transparent;
}
.session-item.active {
	border-color: rgb(56, 189, 248);
	background: rgb(30, 58, 138);
}
.session-title {
	white-space: nowrap;
	overflow: hidden;
	text-overflow: ellipsis;
	max-width: 190px;
}
.delete-btn {
	background: transparent;
	border: none;
	color: rgb(148, 163, 184);
	cursor: pointer;
	font-size: 14px;
	padding: 2px 6px;
}
.delete-btn:hover { color: rgb(248, 113, 113); }
.status-box {
	font-size: 12px;
	color: rgb(148, 163, 184);
	padding-top: 12px;
	border-top: 1px solid rgb(51, 65, 85);
	line-height: 1.5;
}
.main {
	flex: 1;
	display: flex;
	flex-direction: column;
	height: 100vh;
}
.chat-header {
	padding: 14px 24px;
	background: rgb(30, 41, 59);
	border-bottom: 1px solid rgb(51, 65, 85);
	display: flex;
	justify-content: space-between;
	align-items: center;
}
.messages {
	flex: 1;
	overflow-y: auto;
	padding: 24px;
	display: flex;
	flex-direction: column;
	gap: 16px;
}
.message {
	max-width: 820px;
	padding: 14px 18px;
	border-radius: 12px;
	line-height: 1.55;
	white-space: pre-wrap;
	word-wrap: break-word;
}
.message.user {
	align-self: flex-end;
	background: rgb(3, 105, 161);
	color: white;
}
.message.assistant {
	align-self: flex-start;
	background: rgb(30, 41, 59);
	border: 1px solid rgb(51, 65, 85);
}
.sources {
	margin-top: 12px;
	padding-top: 10px;
	border-top: 1px solid rgb(51, 65, 85);
	font-size: 12px;
	display: flex;
	flex-direction: column;
	gap: 4px;
}
.source-link {
	color: rgb(56, 189, 248);
	text-decoration: none;
}
.source-link:hover { text-decoration: underline; }
.input-bar {
	padding: 16px 24px;
	background: rgb(30, 41, 59);
	border-top: 1px solid rgb(51, 65, 85);
	display: flex;
	gap: 12px;
}
.input-bar input {
	flex: 1;
	padding: 12px 16px;
	border-radius: 8px;
	border: 1px solid rgb(71, 85, 105);
	background: rgb(15, 23, 42);
	color: white;
	font-size: 15px;
}
.input-bar button {
	padding: 12px 22px;
	background: rgb(14, 165, 233);
	color: white;
	border: none;
	border-radius: 8px;
	font-weight: 600;
	cursor: pointer;
}
.input-bar button:disabled {
	opacity: 0.5;
	cursor: not-allowed;
}
</style>
</head>
<body>
<div class="sidebar">
	<div class="brand">FastAPI Hybrid RAG</div>
	<button class="new-chat-btn" onclick="createNewSession()">+ Новая сессия</button>
	<div class="sessions-list" id="sessionsList"></div>
	<div class="status-box" id="statusBox">Загрузка статуса...</div>
</div>
<div class="main">
	<div class="chat-header">
		<div id="currentSessionTitle">Выберите или создайте сессию</div>
		<div id="sessionBadge" style="font-size: 12px; color: rgb(148, 163, 184);"></div>
	</div>
	<div class="messages" id="messagesContainer"></div>
	<form class="input-bar" onsubmit="sendMessage(event)">
		<input type="text" id="queryInput" placeholder="Задайте вопрос по документации FastAPI..." autocomplete="off" required />
		<button type="submit" id="sendBtn">Отправить</button>
	</form>
</div>
<script>
let activeSessionId = null;

async function loadStatus() {
	try {
		const res = await fetch('/api/health');
		const data = await res.json();
		document.getElementById('statusBox').innerHTML =
			`Device: <b>${data.device}</b><br>` +
			`Chunks: <b>${data.indexed_chunks}</b><br>` +
			`Dense: ${data.embedding_model}<br>` +
			`Reranker: ${data.reranker_model}`;
	} catch (e) {
		document.getElementById('statusBox').innerText = 'Ошибка подключения';
	}
}

async function loadSessions() {
	const res = await fetch('/api/sessions');
	const sessions = await res.json();
	const container = document.getElementById('sessionsList');
	container.innerHTML = '';
	sessions.forEach(s => {
		const div = document.createElement('div');
		div.className = 'session-item' + (s.id === activeSessionId ? ' active' : '');
		div.onclick = () => selectSession(s.id);
		const titleSpan = document.createElement('span');
		titleSpan.className = 'session-title';
		titleSpan.textContent = s.title || 'New Chat';
		const delBtn = document.createElement('button');
		delBtn.className = 'delete-btn';
		delBtn.textContent = '✕';
		delBtn.onclick = (ev) => { ev.stopPropagation(); deleteSession(s.id); };
		div.appendChild(titleSpan);
		div.appendChild(delBtn);
		container.appendChild(div);
	});
	if (!activeSessionId && sessions.length > 0) {
		await selectSession(sessions[0].id);
	} else if (sessions.length === 0) {
		await createNewSession();
	}
}

async function createNewSession() {
	const res = await fetch('/api/sessions', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ title: 'New Chat' })
	});
	const s = await res.json();
	activeSessionId = s.id;
	await loadSessions();
	await selectSession(s.id);
}

async function deleteSession(sessionId) {
	await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
	if (activeSessionId === sessionId) {
		activeSessionId = null;
	}
	await loadSessions();
}

async function selectSession(sessionId) {
	activeSessionId = sessionId;
	const res = await fetch(`/api/sessions/${sessionId}`);
	if (!res.ok) return;
	const detail = await res.json();
	document.getElementById('currentSessionTitle').textContent = detail.title;
	document.getElementById('sessionBadge').textContent = `Session: ${detail.id.slice(0, 8)}`;
	const container = document.getElementById('messagesContainer');
	container.innerHTML = '';
	detail.messages.forEach(m => appendMessageElement(m.role, m.content, m.sources, m.metrics));
	await loadSessionsWithoutSelect();
}

async function loadSessionsWithoutSelect() {
	const res = await fetch('/api/sessions');
	const sessions = await res.json();
	const container = document.getElementById('sessionsList');
	container.innerHTML = '';
	sessions.forEach(s => {
		const div = document.createElement('div');
		div.className = 'session-item' + (s.id === activeSessionId ? ' active' : '');
		div.onclick = () => selectSession(s.id);
		const titleSpan = document.createElement('span');
		titleSpan.className = 'session-title';
		titleSpan.textContent = s.title || 'New Chat';
		const delBtn = document.createElement('button');
		delBtn.className = 'delete-btn';
		delBtn.textContent = '✕';
		delBtn.onclick = (ev) => { ev.stopPropagation(); deleteSession(s.id); };
		div.appendChild(titleSpan);
		div.appendChild(delBtn);
		container.appendChild(div);
	});
}

function appendMessageElement(role, content, sources = [], metrics = {}) {
	const container = document.getElementById('messagesContainer');
	const div = document.createElement('div');
	div.className = `message ${role}`;
	const textNode = document.createElement('div');
	textNode.textContent = content;
	div.appendChild(textNode);

	if (role === 'assistant' && sources && sources.length > 0) {
		const srcBox = document.createElement('div');
		srcBox.className = 'sources';
		const header = document.createElement('div');
		const lat = metrics && metrics.latency_seconds ? ` (${metrics.latency_seconds}s)` : '';
		header.innerHTML = `<b>Источники FastAPI Reference${lat}:</b>`;
		srcBox.appendChild(header);
		sources.forEach((src, i) => {
			const link = document.createElement('a');
			link.className = 'source-link';
			link.href = src.section_url || '';
			link.target = '_blank';
			const rScore = src.rerank_score !== undefined && src.rerank_score !== null ? ` [rerank: ${Number(src.rerank_score).toFixed(2)}]` : '';
			link.textContent = `${i + 1}. ${src.breadcrumb || src.section_title || 'FastAPI'}${rScore}`;
			srcBox.appendChild(link);
		});
		div.appendChild(srcBox);
	}
	container.appendChild(div);
	container.scrollTop = container.scrollHeight;
}

async function sendMessage(e) {
	e.preventDefault();
	const input = document.getElementById('queryInput');
	const btn = document.getElementById('sendBtn');
	const query = input.value.trim();
	if (!query) return;

	input.value = '';
	btn.disabled = true;
	appendMessageElement('user', query);

	try {
		const res = await fetch('/api/chat', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				query: query,
				session_id: activeSessionId,
				top_k: 5,
				rerank: true
			})
		});
		const data = await res.json();
		activeSessionId = data.session_id;
		appendMessageElement('assistant', data.answer, data.sources, {
			...data.metrics,
			latency_seconds: data.latency_seconds
		});
		await loadSessionsWithoutSelect();
	} catch (err) {
		appendMessageElement('assistant', 'Ошибка при получении ответа от сервера.');
	} finally {
		btn.disabled = false;
		input.focus();
	}
}

loadStatus();
loadSessions();
</script>
</body>
</html>
"""
