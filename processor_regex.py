import re 

def classify_with_regex(log_message):
    regex_patterns = {
        r"User User\d+ logged (in|out)": "User Action",
        r"Backup (started|ended) at .*": "System Notification",
        r"Backup completed successfully": "System Notification",
        r"System updated to version .*": "System Notification",
        r"File .* uploaded successfully by user .*": "System Notification",
        r"Disk cleanup completed successfully.": "System Notification",
        r"System reboot intiated by user .*": "System Notification",
        r"Account with ID .* created by .*": "User Action"

    }

    for pattern, label in regex_patterns.items():
        if re.search(pattern, log_message, re.IGNORECASE):
            return label
    return None

if __name__ == "__main__":
    # Test the function with some log messages
    test_messages = [
        "User User123 logged in",
        "Backup started at 2023-10-01 12:00:00",
        "Backup completed successfully",
        "System updated to version 1.2.3",
        "File report.pdf uploaded successfully by user User456",
        "Disk cleanup completed successfully.",
        "System reboot intiated by user User789",
        "Account with ID 12345 created by admin"
    ]

    for message in test_messages:
        print(f"Log Message: {message} => Classified as: {classify_with_regex(message)}")