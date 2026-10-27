"""Single-page HTML/JS interface for the Multi-Language FastAPI Hybrid RAG Chat."""

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
	width: 290px;
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
.lang-box {
	margin-bottom: 12px;
	display: flex;
	flex-direction: column;
	gap: 5px;
}
.lang-label {
	font-size: 12px;
	color: rgb(148, 163, 184);
	font-weight: 600;
}
.lang-select {
	width: 100%;
	padding: 8px 10px;
	border-radius: 8px;
	border: 1px solid rgb(71, 85, 105);
	background: rgb(15, 23, 42);
	color: rgb(226, 232, 240);
	font-size: 13.5px;
	cursor: pointer;
}
.new-chat-btn {
	background: rgb(14, 165, 233);
	color: white;
	border: none;
	padding: 10px 14px;
	border-radius: 8px;
	font-weight: 600;
	cursor: pointer;
	margin-bottom: 14px;
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
	max-width: 170px;
}
.lang-pill {
	font-size: 10px;
	background: rgb(51, 65, 85);
	color: rgb(186, 230, 253);
	padding: 2px 6px;
	border-radius: 4px;
	text-transform: uppercase;
	margin-left: 6px;
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
	gap: 18px;
}
.message {
	max-width: 860px;
	padding: 16px 20px;
	border-radius: 12px;
	line-height: 1.6;
	word-wrap: break-word;
}
.message.user {
	align-self: flex-end;
	background: rgb(3, 105, 161);
	color: white;
	white-space: pre-wrap;
}
.message.assistant {
	align-self: flex-start;
	background: rgb(30, 41, 59);
	border: 1px solid rgb(51, 65, 85);
	width: 100%;
}
.md-content h1, .md-content h2, .md-content h3 {
	color: rgb(125, 211, 252);
	margin: 12px 0 6px 0;
	font-weight: 600;
}
.md-content h3 { font-size: 16px; }
.md-content h2 { font-size: 17px; }
.md-content p { margin: 8px 0; }
.md-content ul { margin: 8px 0 8px 22px; }
.md-content li { margin: 4px 0; }
.md-content hr {
	border: none;
	border-top: 1px solid rgb(51, 65, 85);
	margin: 16px 0;
}
.md-content a {
	color: rgb(56, 189, 248);
	text-decoration: none;
	font-weight: 500;
}
.md-content a:hover { text-decoration: underline; }
.inline-code {
	background: rgb(15, 23, 42);
	color: rgb(248, 250, 252);
	padding: 2px 6px;
	border-radius: 4px;
	font-family: Consolas, Monaco, monospace;
	font-size: 13px;
	border: 1px solid rgb(51, 65, 85);
}
.code-block {
	margin: 10px 0;
	background: rgb(15, 23, 42);
	border: 1px solid rgb(51, 65, 85);
	border-radius: 8px;
	overflow: hidden;
}
.code-header {
	background: rgb(22, 32, 50);
	padding: 6px 12px;
	font-size: 11px;
	color: rgb(148, 163, 184);
	text-transform: uppercase;
	letter-spacing: 0.5px;
	border-bottom: 1px solid rgb(51, 65, 85);
}
.code-block pre {
	padding: 12px 14px;
	overflow-x: auto;
	font-family: Consolas, Monaco, monospace;
	font-size: 13.5px;
	line-height: 1.5;
	color: rgb(226, 232, 240);
}
.sources {
	margin-top: 16px;
	padding-top: 12px;
	border-top: 1px solid rgb(51, 65, 85);
	font-size: 12.5px;
	display: flex;
	flex-direction: column;
	gap: 6px;
}
.sources-title {
	color: rgb(148, 163, 184);
	font-weight: 600;
	margin-bottom: 2px;
}
.source-badges {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}
.source-link {
	display: inline-flex;
	align-items: center;
	gap: 6px;
	background: rgb(15, 23, 42);
	border: 1px solid rgb(51, 65, 85);
	padding: 5px 10px;
	border-radius: 6px;
	color: rgb(56, 189, 248);
	text-decoration: none;
	font-size: 12px;
}
.source-link:hover {
	border-color: rgb(56, 189, 248);
	background: rgb(23, 37, 84);
}
.score-pill {
	color: rgb(148, 163, 184);
	font-size: 11px;
}
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
	<div class="lang-box">
		<label class="lang-label" for="languageSelect">Язык документации:</label>
		<select id="languageSelect" class="lang-select" onchange="onLanguageChange()">
			<option value="en">en - English</option>
			<option value="de">de - Deutsch</option>
			<option value="es">es - español</option>
			<option value="fr">fr - français</option>
			<option value="hi">hi - हिन्दी</option>
			<option value="ja">ja - 日本語</option>
			<option value="ko">ko - 한국어</option>
			<option value="pt">pt - português</option>
			<option value="ru" selected>ru - русский язык</option>
			<option value="tr">tr - Türkçe</option>
			<option value="uk">uk - українська мова</option>
			<option value="zh">zh - 简体中文</option>
			<option value="zh-hant">zh-hant - 繁體中文</option>
		</select>
	</div>
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

function getSelectedLanguage() {
	const sel = document.getElementById('languageSelect');
	return sel ? sel.value : 'ru';
}

function onLanguageChange() {
	const lang = getSelectedLanguage();
	const badge = document.getElementById('sessionBadge');
	if (activeSessionId && badge) {
		badge.textContent = `Session: ${activeSessionId.slice(0, 8)} · Lang: ${lang.toUpperCase()}`;
	}
}

function escapeHtml(str) {
	return String(str)
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;');
}

function formatInlineMarkdown(text) {
	let s = escapeHtml(text);
	s = s.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
	s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
	s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>');
	s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
	s = s.replace(/(^|\s)(https?:\/\/[^\s<]+)/g, '$1<a href="$2" target="_blank">$2</a>');
	return s;
}

function renderMarkdown(rawMd) {
	if (!rawMd) return '';
	const codeBlocks = [];
	let text = String(rawMd).replace(/\r\n/g, '\n');

	text = text.replace(/^\*\(Note:.*?LLM_API_KEY.*?\)\*\s*/i, '');

	text = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
		const idx = codeBlocks.length;
		const label = lang || 'python';
		codeBlocks.push(
			`<div class="code-block"><div class="code-header">${escapeHtml(label)}</div><pre><code>${escapeHtml(code.trim())}</code></pre></div>`
		);
		return `@@CODEBLOCK_${idx}@@`;
	});

	const lines = text.split('\n');
	const htmlParts = [];
	let inList = false;

	for (let i = 0; i < lines.length; i++) {
		const rawLine = lines[i];
		const trimmed = rawLine.trim();

		if (!trimmed) {
			if (inList) {
				htmlParts.push('</ul>');
				inList = false;
			}
			continue;
		}

		if (trimmed.startsWith('[') && trimmed.endsWith(']') && trimmed.includes(' > ')) {
			continue;
		}
		if (trimmed.startsWith('[Source code in ') && trimmed.endsWith(']')) {
			continue;
		}

		const codeMatch = trimmed.match(/^@@CODEBLOCK_(\d+)@@$/);
		if (codeMatch) {
			if (inList) {
				htmlParts.push('</ul>');
				inList = false;
			}
			htmlParts.push(codeBlocks[Number(codeMatch[1])]);
			continue;
		}

		if (/^-{3,}$/.test(trimmed)) {
			if (inList) {
				htmlParts.push('</ul>');
				inList = false;
			}
			htmlParts.push('<hr>');
			continue;
		}

		const hMatch = trimmed.match(/^(\x23{1,4})\s+(.+)$/);
		if (hMatch) {
			if (inList) {
				htmlParts.push('</ul>');
				inList = false;
			}
			const level = Math.min(hMatch[1].length, 3);
			htmlParts.push(`<h${level}>${formatInlineMarkdown(hMatch[2])}</h${level}>`);
			continue;
		}

		const liMatch = trimmed.match(/^[-*]\s+(.+)$/);
		if (liMatch) {
			if (!inList) {
				htmlParts.push('<ul>');
				inList = true;
			}
			htmlParts.push(`<li>${formatInlineMarkdown(liMatch[1])}</li>`);
			continue;
		}

		if (inList) {
			htmlParts.push('</ul>');
			inList = false;
		}
		htmlParts.push(`<p>${formatInlineMarkdown(trimmed)}</p>`);
	}

	if (inList) {
		htmlParts.push('</ul>');
	}

	return htmlParts.join('');
}

async function loadStatus() {
	try {
		const res = await fetch('/api/health');
		const data = await res.json();
		const langCount = data.languages ? data.languages.length : 13;
		document.getElementById('statusBox').innerHTML =
			`Device: <b>${data.device}</b><br>` +
			`Chunks: <b>${data.indexed_chunks}</b> (${langCount} langs)<br>` +
			`LLM: <b>${data.llm_model}</b><br>` +
			`Dense: ${data.embedding_model}`;
	} catch (e) {
		document.getElementById('statusBox').innerText = 'Ошибка подключения';
	}
}

function renderSessionItem(s, container) {
	const div = document.createElement('div');
	div.className = 'session-item' + (s.id === activeSessionId ? ' active' : '');
	div.onclick = () => selectSession(s.id);
	const leftWrap = document.createElement('div');
	leftWrap.style.display = 'flex';
	leftWrap.style.alignItems = 'center';
	const titleSpan = document.createElement('span');
	titleSpan.className = 'session-title';
	titleSpan.textContent = s.title || 'New Chat';
	const langPill = document.createElement('span');
	langPill.className = 'lang-pill';
	langPill.textContent = s.language || 'ru';
	leftWrap.appendChild(titleSpan);
	leftWrap.appendChild(langPill);

	const delBtn = document.createElement('button');
	delBtn.className = 'delete-btn';
	delBtn.textContent = '✕';
	delBtn.onclick = (ev) => { ev.stopPropagation(); deleteSession(s.id); };
	div.appendChild(leftWrap);
	div.appendChild(delBtn);
	container.appendChild(div);
}

async function loadSessions() {
	const res = await fetch('/api/sessions');
	const sessions = await res.json();
	const container = document.getElementById('sessionsList');
	container.innerHTML = '';
	sessions.forEach(s => renderSessionItem(s, container));
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
		body: JSON.stringify({
			title: 'New Chat',
			language: getSelectedLanguage()
		})
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
	if (detail.language) {
		document.getElementById('languageSelect').value = detail.language;
	}
	document.getElementById('currentSessionTitle').textContent = detail.title;
	document.getElementById('sessionBadge').textContent =
		`Session: ${detail.id.slice(0, 8)} · Lang: ${(detail.language || 'ru').toUpperCase()}`;
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
	sessions.forEach(s => renderSessionItem(s, container));
}

function appendMessageElement(role, content, sources = [], metrics = {}) {
	const container = document.getElementById('messagesContainer');
	const div = document.createElement('div');
	div.className = `message ${role}`;

	if (role === 'assistant') {
		const contentNode = document.createElement('div');
		contentNode.className = 'md-content';
		contentNode.innerHTML = renderMarkdown(content);
		div.appendChild(contentNode);
	} else {
		const textNode = document.createElement('div');
		textNode.textContent = content;
		div.appendChild(textNode);
	}

	if (role === 'assistant' && sources && sources.length > 0) {
		const srcBox = document.createElement('div');
		srcBox.className = 'sources';
		const header = document.createElement('div');
		header.className = 'sources-title';
		const lat = metrics && metrics.latency_seconds ? ` · ${metrics.latency_seconds}s` : '';
		header.textContent = `Источники FastAPI Reference${lat}`;
		srcBox.appendChild(header);

		const badgeRow = document.createElement('div');
		badgeRow.className = 'source-badges';
		sources.forEach((src, i) => {
			const link = document.createElement('a');
			link.className = 'source-link';
			link.href = src.section_url || '';
			link.target = '_blank';
			const titleSpan = document.createElement('span');
			titleSpan.textContent = `${i + 1}. ${src.breadcrumb || src.section_title || 'FastAPI'}`;
			link.appendChild(titleSpan);
			if (src.rerank_score !== undefined && src.rerank_score !== null) {
				const scoreSpan = document.createElement('span');
				scoreSpan.className = 'score-pill';
				scoreSpan.textContent = `(${Number(src.rerank_score).toFixed(2)})`;
				link.appendChild(scoreSpan);
			}
			badgeRow.appendChild(link);
		});
		srcBox.appendChild(badgeRow);
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
				language: getSelectedLanguage(),
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
