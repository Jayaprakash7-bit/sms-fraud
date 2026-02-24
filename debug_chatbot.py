import traceback
from src.sms_fraud import chatbot

print('chatbot module:', chatbot)

try:
    print('Calling get_response_local("hello")')
    resp = chatbot.get_response_local('hello')
    print('Response:', repr(resp))
except Exception:
    print('Exception occurred:')
    traceback.print_exc()