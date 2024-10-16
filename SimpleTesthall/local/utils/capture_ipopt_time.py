import sys
import io
import re


class DualOutput:
    """
    Write the outputs to both the console and a StringIO buffer.
    """

    def __init__(self, *streams):
        self.streams = streams  # Streams: Console (sys.stdout) and StringIO buffer

    def write(self, data):
        for stream in self.streams:
            stream.write(data)  # Write data to all streams

    def flush(self):
        for stream in self.streams:
            stream.flush()  # Ensure all content is written


class IpoptOutputCapture:
    """
    Capture and process IPOPT output from the console.
    """

    def __init__(self):
        self.buffer = io.StringIO()  # Memory buffer for captured output
        self.original_stdout = sys.stdout  # Save the original stdout

    def start_capture(self):
        sys.stdout = DualOutput(self.original_stdout, self.buffer)  # Redirect stdout to console and buffer

    def stop_capture(self):
        sys.stdout = self.original_stdout  # Restore original stdout
        return self.buffer.getvalue()  # Return captured output

    @staticmethod
    def process_ipopt_times(captured_output):
        """
        Extract and calculate total and average IPOPT times from captured output.
        """
        ipopt_times = re.findall(r"Total seconds in IPOPT\s*=\s*(\d+\.\d+)", captured_output)
        ipopt_times = [float(time) for time in ipopt_times]

        if ipopt_times:
            total_ipopt_time = sum(ipopt_times)
            average_ipopt_time = total_ipopt_time / len(ipopt_times)
            print(f"Total IPOPT time: {total_ipopt_time} seconds")
            print(f"Average IPOPT time: {average_ipopt_time} seconds")
        else:
            print("No IPOPT times were captured")