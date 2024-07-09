document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('.signup-form');

    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent form submission

        // Get form values
        const username = document.getElementById('username').value.trim();
        const email = document.getElementById('email').value.trim();
        const password = document.getElementById('password').value.trim();
        const confirmPassword = document.getElementById('confirm-password').value.trim();
        const allergies = document.getElementById('allergies').value.trim();
        const foodPreference = document.getElementById('food-preference').value;

        // Validate form values
        if (!username || !email || !password || !confirmPassword) {
            alert('Please fill in all required fields.');
            return;
        }

        if (password !== confirmPassword) {
            alert('Passwords do not match.');
            return;
        }

        // If everything is valid, you can proceed with form submission
        const formData = {
            username: username,
            email: email,
            password: password,
            allergies: allergies,
            foodPreference: foodPreference,
        };

        console.log('Form Data:', formData);

        // For demonstration, we are directly redirecting to the home page after validation
        alert('Sign up successful!');
        window.location.href = '/first'; // Redirect to home page
    });
});
