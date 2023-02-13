export async function markdownToHTML(markdown_text) {
    var myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");

    var raw = JSON.stringify({
        "mode": "markdown",
        "text": "# Markdown\nRocks!"
    });

    var requestOptions = {
        method: 'POST',
        headers: myHeaders,
        body: raw,
        redirect: 'follow'
    };

    return await fetch("https://api.github.com/markdown", requestOptions)
    .then(response => response.text())
    .then(result => {
        console.log(result)
        return result
    })
    .catch(error => console.log('error', error));
}