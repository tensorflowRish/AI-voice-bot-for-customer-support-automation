responses = {

"order_status":
"Please provide your order ID to check the status.",

"cancel_order":
"Your cancellation request has been received.",

"refund_request":
"Your refund request has been initiated.",

"payment_issue":
"It seems there is a payment issue. Please retry or contact support.",

"change_address":
"You can update your address from your profile settings.",

"delivery_delay":
"We apologize for the delay. Your order will arrive soon.",

"complaint":
"We have registered your complaint.",

"speak_to_agent":
"I will connect you with a support agent."
}

def generate_response(intent):

    return responses.get(intent,"Sorry, I could not understand your request.")