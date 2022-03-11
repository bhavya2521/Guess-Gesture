const signUpButton = document.getElementById('new');
const signInButton = document.getElementById('old');
const container = document.getElementById('container');

const signUp = document.getElementById('signup');
const signIn = document.getElementById('signin');
const feedback = document.getElementById('feedback');



signUpButton.addEventListener('click', () => {
	container.classList.add("right-panel-active");
});

signInButton.addEventListener('click', () => {
	container.classList.remove("right-panel-active");
});


signIn.addEventListener('click', () => {
window.open("home.html");
});

feedback.addEventListener('click', () => {
window.open("index.html");
});






