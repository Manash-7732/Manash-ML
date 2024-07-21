import json
import random
from docxtpl import DocxTemplate

def load_items(json_filename):
    with open(json_filename, 'r') as file:
        return json.load(file)

def create_invoice_list(items):
    # Define fixed prices for each item
    prices = {
        "apple": 20,
        "parle g": 10,
        "lays": 15,
        "banana": 5,
        "dairy milk": 30
    }

    invoice_list = []
    for label_name in items:
        quantity = random.randint(1, 10)  # Generate a random quantity between 1 and 10
        price = prices.get(label_name.lower(), 0)  # Get price from the dictionary, default to 0 if not found
        total_price = price * quantity
        invoice_list.append([quantity, label_name, price, total_price])
    return invoice_list

def generate_invoice(invoice_list):
    doc = DocxTemplate("Invoice_Template.docx")
    subtotal = sum(item[3] for item in invoice_list)
    salestax = subtotal * 0.1  # Assuming a 10% sales tax
    total = subtotal + salestax
    context = {
        "name": "Likith",
        "phone": "1234567890",
        "invoice_list": invoice_list,
        "subtotal": subtotal,
        "salestax": f"{salestax:.2f}",
        "total": f"{total:.2f}"
    }
    doc.render(context)
    doc.save("Generated_Invoice.docx")

# Main execution
items = load_items("detected_items.json")
invoice_list = create_invoice_list(items)
print(invoice_list)
generate_invoice(invoice_list)
