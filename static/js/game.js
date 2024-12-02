// Variables for game elements and state
let score = 0;
let fishElement, heartElement, scoreDisplay, gameArea;

// Function to initialize game elements after DOM loads
function docReady() {
    // Retrieve elements
    fishElement = document.getElementById("fish");
    heartElement = document.getElementById("heart");
    scoreDisplay = document.getElementById("current-score");
    gameArea = document.getElementById("game-area");

    // Event listener for fish movement (you can change it to a more complex movement system)
    gameArea.addEventListener("mousemove", moveFish);
    // Optional: Use keyboard arrow keys to control fish movement
    document.addEventListener("keydown", moveFishWithKeyboard);
    
    // Initialize game state
    resetGame();
}

// Function to move the fish to where the mouse is located
function moveFish(event) {
    // Offset the fish position to center it around the cursor
    const offsetX = event.offsetX - fishElement.clientWidth / 2;
    const offsetY = event.offsetY - fishElement.clientHeight / 2;
    fishElement.style.left = `${offsetX}px`;
    fishElement.style.top = `${offsetY}px`;

    // Check for heart interaction
    detectHeartCollision();
}

// Optional function to allow keyboard movement of the fish
function moveFishWithKeyboard(event) {
    const fishStep = 20; // Step distance for each key press
    let fishX = fishElement.offsetLeft;
    let fishY = fishElement.offsetTop;

    switch (event.key) {
        case "ArrowUp":
            fishY -= fishStep;
            break;
        case "ArrowDown":
            fishY += fishStep;
            break;
        case "ArrowLeft":
            fishX -= fishStep;
            break;
        case "ArrowRight":
            fishX += fishStep;
            break;
    }

    fishElement.style.left = `${fishX}px`;
    fishElement.style.top = `${fishY}px`;

    detectHeartCollision();
}

// Detects if fish and heart are close enough for a "kiss"
function detectHeartCollision() {
    const fishRect = fishElement.getBoundingClientRect();
    const heartRect = heartElement.getBoundingClientRect();

    const overlap = !(
        fishRect.right < heartRect.left ||
        fishRect.left > heartRect.right ||
        fishRect.bottom < heartRect.top ||
        fishRect.top > heartRect.bottom
    );

    if (overlap) {
        handleKiss();
    }
}

// Handles scoring and shows heart animation
function handleKiss() {
    // Increment score
    score += 1;
    scoreDisplay.textContent = score;

    // Show heart briefly
    heartElement.classList.remove("hidden");
    setTimeout(() => {
        heartElement.classList.add("hidden");
    }, 500); // Heart disappears after 500 ms

    // Randomize heart position
    positionHeartRandomly();
}

// Function to start the game
function startGame() {
    score = 0;
    scoreDisplay.textContent = score;
    positionHeartRandomly();
}

// Function to reset the game state
function resetGame() {
    score = 0;
    scoreDisplay.textContent = score;
    positionHeartRandomly();
    heartElement.classList.add("hidden"); // Hide heart initially
}

// Places the heart at a random position within the game area
function positionHeartRandomly() {
    const maxX = gameArea.clientWidth - heartElement.clientWidth;
    const maxY = gameArea.clientHeight - heartElement.clientHeight;

    const randomX = Math.floor(Math.random() * maxX);
    const randomY = Math.floor(Math.random() * maxY);

    heartElement.style.left = `${randomX}px`;
    heartElement.style.top = `${randomY}px`;
}
