from flask import Flask, request, render_template
import datetime
import inspect
app = Flask(__name__,template_folder='templates_new',static_folder='static_new')
# Logging function
def logit(message):
    try:
        # Get the current timestamp
        timestr = datetime.datetime.now().strftime('%A_%b-%d-%Y_%H-%M-%S')
        print(f"timestr: {timestr}")

        # Get the caller's frame information
        caller_frame = inspect.stack()[1]
        filename = caller_frame.filename
        lineno = caller_frame.lineno

        # Convert message to string if it's a list
        if isinstance(message, list):
            message_str = ' '.join(map(str, message))
        else:
            message_str = str(message)

        # Construct the log message with filename and line number
        log_message = f"{timestr} - File: {filename}, Line: {lineno}: {message_str}\n"

        # Open the log file in append mode
        with open("upload_log.txt", "a") as file:
            # Write the log message to the file
            file.write(log_message)

            # Print the log message to the console
            print(log_message)

    except Exception as e:
        # If an exception occurs during logging, print an error message
        print(f"Error occurred while logging: {e}")

logit("App started")
def readlog():
    log_file_path = 'upload_log.txt'    
    with open(log_file_path, "r") as Input:
        logdata = Input.read()
    # print last entry
    logdata = logdata.split("\n")
    return logdata

logdata = readlog()
logit("This is a DEBUG message for mylog.py" + str(logdata))

@app.route('/view_log', methods=['GET', 'POST'])
def view_log():
    data = readlog()
    return render_template('view_log.html', data=data)


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/upload_mp4_video')
def upload_mp4_video():
    return render_template('upload_mp4.html')
    
@app.route('/upload_mp4', methods=['POST'])
def upload_mp4():
    uploaded_file = request.files['videoFile']
    if uploaded_file.filename != '':
        # Save the uploaded file to a directory or process it as needed
        # For example, you can save it to a specific directory:
        uploaded_file.save('static/video_resources/use.mp4')
        #                   /' + uploaded_file.filename)
        VIDEO='static/video_resources/use.mp4'
        return render_template('upload_mp4.html',VIDEO=VIDEO)
    else:
        VIDEO='static/video_resources/use.mp4'
        return render_template('upload_mp4.html',VIDEO=VIDEO)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5100)
