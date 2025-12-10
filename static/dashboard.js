async function fetchJSON(url) {
    const res = await fetch(url)
    return await res.json()
}

async function loadDashboard() {
    const dailyCount = await fetchJSON("/api/analytics/daily_count");
    const topChunks = await fetchJSON("/api/analytics/top_chunks");
    const topQuestions = await fetchJSON("/api/analytics/top_questions")
    const answerLength = await fetchJSON("/api/analytics/answer_length");

    // Cards ----------------------------------------------
    document.getElementById("totalQuestions").innerText = Object.values(dailyCount).reduce((a,b) => a + b, 0);
    document.getElementById("avgAnswerLength").innerText = Math.round(answerLength.avg);
    document.getElementById("topChunk").innerText = topChunks.length ? topChunks[0][0]: "-";
    document.getElementById("topQuestion").innerHTML = topQuestions.length ? topQuestions[0][0]: "-";

    // Charts ---------------------------------------------
    new Chart(document.getElementById("dailyChart"), {
        type: "bar",
        data: {
            labels: Object.keys(dailyCount),
            datasets: [{
                label: "Questions per day",
                data: Object.values(dailyCount),
                borderWidth: 2
            }]
        }
    })

    new Chart(document.getElementById("dailyChartLine"), {
        type: "line",
        data: {
            labels: Object.keys(dailyCount),
            datasets: [{
                label: "Questions per day",
                data: Object.values(dailyCount),
                borderWidth: 2
            }]
        }
    })

    new Chart(document.getElementById("chunkChart"), {
        type: "bar",
        data: {
            labels: topChunks.map(t => t[0]),
            datasets: [{
                label: "Chunks usage count",
                data: topChunks.map(t => t[1]),
                borderWidth: 2
            }]
        }
    })

    new Chart(document.getElementById("questionChart"), {
        type: "bar",
        data: {
            labels: topQuestions.map(t => t[0]),
            datasets: [{
                label: "Frequency",
                data: topQuestions.map(t => t[1]),
                borderWidth: 2
            }]
        }
    })
}

loadDashboard()