window.onload = function () {
    const video = document.getElementById("videoPlayer");
    const startTimeSlider = document.getElementById("startTimeSlider");
    const endTimeSlider = document.getElementById("endTimeSlider");

    // Adjust the slider max value to the video's duration once the video is loaded
    video.onloadedmetadata = function () {
        startTimeSlider.max = video.duration;
        endTimeSlider.max = video.duration;
        endTimeSlider.value = video.duration;
    };

    // Change video start time when the start time slider is moved
    startTimeSlider.addEventListener("input", function () {
        video.currentTime = startTimeSlider.value;
    });

    // Pause video when the current time reaches the end time slider value
    video.addEventListener("timeupdate", function () {
        if (video.currentTime >= endTimeSlider.value) {
            video.pause();
        }
    });

    // Apply the start/stop edits when the button is clicked
    document.getElementById("applyEditsButton").addEventListener("click", function () {
        const startTime = startTimeSlider.value;
        const endTime = endTimeSlider.value;
        
        // Send start and end time to Flask backend for trimming
        fetch('/trim-video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ start: startTime, end: endTime })
        }).then(response => response.json()).then(data => {
            console.log('Video trimmed successfully:', data);
        }).catch(error => {
            console.error('Error trimming video:', error);
        });
    });
};
