## Using the Live Streaming Transcript From AssemblyAI & LeMUR to Get Clips

```process_stream.py```
- add a file to /files
- copy that file's path and add set the input_filename variable (line 71 in process_stream.py) 
- run process_stream.py to create the equivalent to the sentences[] list you can export from an async transcription
    - you'll get the text from the entire transcript (this is done by just accumulating final transcripts throughout the stream)
    - you'll get a sentences list (which splits the transcript into a series of final transcripts - each with a start and end timestamp)

```get_clips.py``` 
- add your own prompt and company name to the lemur logic
- run that code to pull out a series of engaging clips that is relevant to the prompt

### Requirements
- run pip install requirements.txt
- assemblyai key
- a file to process (put this file in the /files folder)

