document.getElementById('emailForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    const senderEmail = document.getElementById('senderEmail').value;
    const emailText = document.getElementById('emailText').value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            senderEmail: senderEmail,
            emailText: emailText
        }),
    });
    const result = await response.json();
    document.getElementById('result').innerText = result.phishing ? 'This email is shady. Be careful clicking on any links in this email' : 'This email is safe.';
});