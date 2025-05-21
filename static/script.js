function summarize() {
    let youtubeUrl = document.getElementById("youtube-url").value;
    
    if (!youtubeUrl) {
        alert("Please enter a YouTube URL.");
        return;
    }

    document.getElementById("summary-text").innerText = "Summarizing... Please wait.";

    fetch("/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: youtubeUrl })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("summary-text").innerText = "Error: " + data.error;
        } else {
            document.getElementById("summary-text").innerText = data.summary;
        }
    })
    .catch(error => {
        document.getElementById("summary-text").innerText = "Error fetching summary.";
        console.error("Error:", error);
    });
}
