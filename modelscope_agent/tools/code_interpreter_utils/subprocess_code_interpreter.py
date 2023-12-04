import queue
import subprocess
import threading
import time
import traceback

from .base_code_interpreter import BaseCodeInterpreter


class SubprocessCodeInterpreter(BaseCodeInterpreter):

    def __init__(self):
        self.start_cmd = ''
        self.process = None
        self.debug_mode = False
        self.output_queue = queue.Queue()
        self.done = threading.Event()

    def detect_active_line(self, line):
        return None

    def detect_end_of_execution(self, line):
        return None

    def line_postprocessor(self, line):
        return line

    def preprocess_code(self, code):
        """
        This needs to insert an end_of_execution marker of some kind,
        which can be detected by detect_end_of_execution.

        Optionally, add active line markers for detect_active_line.
        """
        return code

    def terminate(self):
        self.process.terminate()

    def start_process(self):
        if self.process:
            self.terminate()

        self.process = subprocess.Popen(
            self.start_cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            universal_newlines=True,
        )
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stdout, False),
            daemon=True,
        ).start()
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stderr, True),
            daemon=True,
        ).start()

    def run(self, code):
        retry_count = 0
        max_retries = 3

        # Setup
        try:
            code = self.preprocess_code(code)
            if not self.process:
                self.start_process()
        except Exception as e:
            print(e)
            yield {'output': traceback.format_exc()}
            return

        while retry_count <= max_retries:
            if self.debug_mode:
                print(
                    f'(after processing) Running processed code:\n{code}\n---')

            self.done.clear()

            try:
                self.process.stdin.write(code + '\n')
                self.process.stdin.flush()
                break
            except Exception as e:
                print(e)
                if retry_count != 0:
                    # For UX, I like to hide this if it happens once. Obviously feels better to not see errors
                    # Most of the time it doesn't matter, but we should figure out why it happens frequently with:
                    # applescript
                    yield {'output': traceback.format_exc()}
                    yield {
                        'output': f'Retrying... ({retry_count}/{max_retries})'
                    }
                    yield {'output': 'Restarting process.'}

                self.start_process()

                retry_count += 1
                if retry_count > max_retries:
                    yield {
                        'output':
                        'Maximum retries reached. Could not execute code.'
                    }
                    return

        while True:
            if not self.output_queue.empty():
                yield self.output_queue.get()
            else:
                time.sleep(0.1)
            try:
                output = self.output_queue.get(
                    timeout=0.3)  # Waits for 0.3 seconds
                yield output
            except queue.Empty:
                if self.done.is_set():
                    # Try to yank 3 more times from it... maybe there's something in there...
                    # (I don't know if this actually helps. Maybe we just need to yank 1 more time)
                    for _ in range(3):
                        if not self.output_queue.empty():
                            yield self.output_queue.get()
                        time.sleep(0.2)
                    break

    def handle_stream_output(self, stream, is_error_stream):
        for line in iter(stream.readline, ''):
            if self.debug_mode:
                print(f'Received output line:\n{line}\n---')

            line = self.line_postprocessor(line)

            if line is None:
                continue  # `line = None` is the postprocessor's signal to discard completely

            if self.detect_active_line(line):
                active_line = self.detect_active_line(line)
                self.output_queue.put({'active_line': active_line})
            elif self.detect_end_of_execution(line):
                self.output_queue.put({'active_line': None})
                time.sleep(0.1)
                self.done.set()
            elif is_error_stream and 'KeyboardInterrupt' in line:
                self.output_queue.put({'output': 'KeyboardInterrupt'})
                time.sleep(0.1)
                self.done.set()
            else:
                self.output_queue.put({'output': line})
