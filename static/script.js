const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const projectList = document.getElementById('project-list');

dropZone.onclick = () => fileInput.click();

fileInput.onchange = async (e) => {
    const files = e.target.files;
    const formData = new FormData();
    for (let file of files) {
        formData.append("files", file);
        createProjectCard(file.name);
    }
    const response = await fetch('/upload', { method: 'POST', body: formData });
    const data = await response.json();
    data.forEach(proj => startPolling(proj.id, proj.filename));
};

function createProjectCard(filename) {
    const safeId = filename.replace(/\W/g, '');
    const cardHtml = `
        <div class="card" id="card-${safeId}">
            <div class="stats">
                <span>${filename}</span>
                <span class="status-tag" id="stage-${safeId}">QUEUED</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="bar-${safeId}"></div>
            </div>
            <div class="stats">
                <span id="perc-${safeId}">0%</span>
            </div>
        </div>`;
    projectList.insertAdjacentHTML('afterbegin', cardHtml);
}

function startPolling(pid, filename) {
    const safeId = filename.replace(/\W/g, '');
    const poller = setInterval(async () => {
        const res = await fetch(`/status/${pid}`);
        const json = await res.json();
        const d = json.data;
        
        document.getElementById(`bar-${safeId}`).style.width = d.percent + '%';
        document.getElementById(`stage-${safeId}`).innerText = d.stage;
        document.getElementById(`perc-${safeId}`).innerText = d.percent + '%';
        
        if (d.percent >= 100 || d.percent === -1) clearInterval(poller);
    }, 1000);
}