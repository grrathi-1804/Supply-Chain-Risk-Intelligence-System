import sys

def error_message_detail(error, error_detail: sys):
    # exc_tb contains the traceback (where the error happened)
    _, _, exc_tb = error_detail.exc_info()
    
    # Extracting the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Properly formatting the string with variables passed into .format()
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Pass the message to the base Exception class
        super().__init__(error_message)
        
        # Call the detail function to get the formatted message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        # This ensures that when you print the exception, it shows the detailed message
        return self.error_message