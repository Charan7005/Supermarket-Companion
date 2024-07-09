import argparse
import csv
import datetime
import json
import logging
import time
import cv2
from flask import Blueprint, Flask, Response, jsonify, request, render_template, session
import mysql.connector
import google.generativeai as genai 
from itertools import combinations, zip_longest
from collections import defaultdict
import numpy as np
from werkzeug.security import generate_password_hash
import threading
from people_counter import generate_frames, get_counts
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject  # Ensure you have a function to get counts
import dlib
from imutils.video import VideoStream
from utils import thread
from utils.mailer import Mailer
import imutils
from imutils.video import FPS
from people_counter import get_counts
app = Flask(__name__, template_folder='templates')
cart_items = []  # This will store items temporarily in memory
app.secret_key = 'supersecretkey'

# Configure the logging
logging.basicConfig(level=logging.DEBUG)

# MySQL configuration
config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'rexchanop@2020',
    'database': 'miniproject'
}

# Function to connect to the MySQL database
def connect_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='rexchanop@2020',
        database='miniproject'
    )

def generate_candidates(itemset, length):
    return set(combinations(itemset, length))

def count_occurrences(transactions, candidate_itemset):
    count = 0
    for transaction in transactions:
        if candidate_itemset.issubset(transaction):
            count += 1
    return count

def apriori(transactions, min_support, min_confidence):
    # Phase 1: Find frequent itemsets
    itemsets = [frozenset([item]) for transaction in transactions for item in transaction]
    frequent_itemsets = []
    length = 1
    while itemsets:
        candidate_itemsets = set()
        count_dict = defaultdict(int)
        
        for itemset in itemsets:
            count_dict[itemset] += 1
        
        for itemset, count in count_dict.items():
            support = count / len(transactions)
            if support >= min_support:
                frequent_itemsets.append((itemset, support))
                candidate_itemsets.update(itemset)
        
        length += 1
        itemsets = generate_candidates(candidate_itemsets, length)
    
    # Phase 2: Generate association rules
    association_rules = []
    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    antecedent_support = count_occurrences(transactions, antecedent) / len(transactions)
                    confidence = support / antecedent_support
                    if confidence >= min_confidence:
                        association_rules.append((antecedent, consequent, confidence))
    
    return frequent_itemsets, association_rules

def get_cart_items():
    global cart_items
    return session.get('cart_items', cart_items)

def clear_cart():
    global cart_items
    cart_items = []
    session.pop('cart_items', None)

# Example transactions (replace with your actual data retrieval from a database)
def load_transactions():
    transactions = []
    for item in get_cart_items():
        transactions.append(frozenset([item['product_name']]))
    return transactions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/first')
def first():
    return render_template('home.html')

@app.route('/second')
def second():
    return render_template('signup.html')

@app.route('/third')
def third():
    return render_template('search.html')

@app.route('/fourth')
def fourth():
    return render_template('aichef.html')

@app.route('/fifth')
def fifth():
    return render_template('barcode.html')

bill = []
@app.route('/six')
def six():
    return render_template('checkout.html', bill=bill)

# Route to render the HTML page showing counts
@app.route('/seven')
def seven():
    entered, exited, inside = get_counts()  # Replace with actual function call
    return render_template('try.html', entered=entered, exited=exited, inside=inside)




def run_people_counter():
    generate_frames()


@app.route('/scan_barcode', methods=['POST'])
def receive_barcode():
    if request.method == 'POST':
        barcode_data = request.json.get('barcode')
        print("Received Barcode:", barcode_data)
        
        # Connect to the database
        conn = connect_db()
        cursor = conn.cursor()

        # Execute a query to retrieve data based on the scanned barcode
        query = "SELECT product_name, price, quantity, lactose_intolerant, vegan, nut_allergic, supermarket_name FROM inventoryone WHERE product_id = %s"
        cursor.execute(query, (barcode_data,))
        result = cursor.fetchone()

        if result:
            product_name = result[0]
            price = float(result[1])
            quantity = result[2]
            lactose_intolerant = result[3]
            vegan = result[4]
            nut_allergic = result[5]
            supermarket_name = result[6]
            cart_items.append({'product_name': product_name, 'price': price, 'quantity': quantity, 'lactose_intolerant': lactose_intolerant, 'vegan': vegan, 'nut_allergic': nut_allergic, 'supermarket_name': supermarket_name})
            session['cart_items'] = cart_items
            
            return jsonify({'status': 'success', 'barcode': barcode_data, 'product_name': product_name, 'price': price, 'quantity': quantity, 'lactose_intolerant': lactose_intolerant, 'vegan': vegan, 'nut_allergic': nut_allergic, 'supermarket_name': supermarket_name})

        else:
            return jsonify({'status': 'error', 'message': 'Barcode not found'})

        conn.close()

    else:
        return jsonify({'error': 'Method not allowed'}), 405

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q')
    if query:
        conn = connect_db()
        cursor = conn.cursor(dictionary=True)

        # Execute a query to search for products based on the search query
        search_query = "SELECT product_name, price, quantity, lactose_intolerant, vegan, nut_allergic, supermarket_name FROM inventoryone WHERE product_name LIKE %s"
        cursor.execute(search_query, ('%' + query + '%',))
        search_results = cursor.fetchall()
        conn.close()

        if search_results:
            return jsonify(search_results)
        else:
            return jsonify([])  # Return an empty list if no results found
    else:                       
        return jsonify([])      # Return an empty list if no query provided


@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    data = request.json
    product_name = data.get('product_name')
    price = data.get('price')
    quantity = data.get('quantity')
    lactose_intolerant = data.get('lactose_intolerant')
    nut_allergic = data.get('nut_allergic')
    vegan = data.get('vegan')

    # Add the item to the cart
    cart_items.append({
        'product_name': product_name,
        'price': price,
        'quantity': quantity,
        'lactose_intolerant': lactose_intolerant,
        'nut_allergic': nut_allergic,
        'vegan': vegan
    })

    response_data = {
        'status': 'success',
        'message': 'Item added to cart',
        'product_name': product_name,
        'price': price,
        'quantity': quantity,
        'lactose_intolerant': lactose_intolerant,
        'nut_allergic': nut_allergic,
        'vegan': vegan
    }
    return jsonify(response_data)


checkout_bp = Blueprint('checkout', __name__)

@checkout_bp.route('/checkout', methods=['GET'])
def checkout():
   # Calculate the total price
    total_price = sum(float(item['price']) for item in cart_items)
    
    bill = {
        'cart_items': cart_items,
        'total_price': total_price
    }
    
    # Return or use 'bill' as needed in your application
    return render_template('checkout.html', bill=bill)

app.register_blueprint(checkout_bp) 

genai.configure(api_key="AIzaSyAF4JDFwMDGF3Fj8_rxvW8KtAhSDE7pSCw")

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    print("User Input:", user_input)  # Log the user input for debugging

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(user_input)
        bot_response = response.text
        print("Bot Response:", bot_response)  # Log the bot response for debugging
        return jsonify({'response': bot_response})
    except Exception as e:
        print("Error:", e)  # Log the error for debugging
        return jsonify({'response': 'Sorry, I am having trouble generating a response at the moment.'})

@app.route('/check_ingredients', methods=['POST'])
def check_ingredients():
    data = request.json
    ingredients = data['ingredients']
    print("Ingredients to check:", ingredients)  # Log the ingredients for debugging
    
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)

    format_strings = ','.join(['%s'] * len(ingredients))
    query = f"SELECT * FROM inventoryone WHERE product_name IN ({format_strings})"
    cursor.execute(query, tuple(ingredients))
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    return jsonify(results)

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    logging.debug(f'Received data: {data}')
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    allergies = data.get('allergies', '')
    food_preference = data.get('food_preference', '')

    if not username or not email or not password:
        logging.warning('Required fields are missing')
        return jsonify({'error': 'Please fill in all required fields.'}), 400

    if '@' not in email or '.' not in email:
        logging.warning('Invalid email address')
        return jsonify({'error': 'Please provide a valid email address.'}), 400

    if len(password) < 6:
        logging.warning('Password is too short')
        return jsonify({'error': 'Password must be at least 6 characters long.'}), 400

    hashed_password = generate_password_hash(password)
    logging.debug('Password hashed successfully')

    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            logging.warning('Email already in use')
            return jsonify({'error': 'Email already in use.'}), 400

        cursor.execute("""
            INSERT INTO Users (username, email, password, allergies, food_preference)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, email, hashed_password, allergies, food_preference))
        conn.commit()
        cursor.close()
        conn.close()
        logging.debug('User registered successfully')
    except mysql.connector.Error as err:
        logging.error(f'Database error: {err}')
        return jsonify({'error': str(err)}), 500

    return jsonify({'message': 'Sign up successful!'}), 201








# Execution start time
start_time = time.time()
# Setup logger
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)
# Initiate features config
with open("utils/config.json", "r") as file:
    config = json.load(file)


total_in = 0
total_out = 0
total_inside = 0





def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True, help="D:\vs code\BarcodeBilling\BarcodeBilling\project1\models\deploy.prototxt")
    ap.add_argument("-m", "--model", required=True, help="D:\vs code\BarcodeBilling\BarcodeBilling\project1\models\res10_300x300_ssd_iter_140000.caffemodel")
    ap.add_argument("-i", "--input", type=str, help="Path to optional input video file")
    ap.add_argument("-o", "--output", type=str, help="Path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4, help="Minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30, help="# of skip frames between detections")
    args = vars(ap.parse_args())
    return args

def send_mail():
    Mailer().send(config["Email_Receive"])

def log_data(move_in, in_time, move_out, out_time):
    data = [move_in, in_time, move_out, out_time]
    export_data = zip_longest(*data, fillvalue='')
    with open('utils/data/logs/counting_data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if myfile.tell() == 0:
            wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
        wr.writerows(export_data)

def generate_frames():
    global total_in, total_out, total_inside
    args = parse_arguments()
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    if not args.get("input", False):
        logger.info("Starting the live stream..")
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
    else:
        logger.info("Starting the video..")
        vs = cv2.VideoCapture(args["input"])

    writer = None
    W = None
    H = None
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    total = []
    move_out = []
    move_in = []
    out_time = []
    in_time = []

    fps = FPS().start()

    if config["Thread"]:
        vs = thread.ThreadingClass(config["url"])

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        if args["input"] is not None and frame is None:
            break

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []

        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        cv2.putText(frame, "-Prediction border - Entrance-", (10, H - 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        total_out += 1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        move_out.append(totalUp)
                        out_time.append(date_time)
                        to.counted = True
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        total_in += 1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        move_in.append(totalDown)
                        in_time.append(date_time)
                        if sum(total) >= config["Threshold"]:
                            cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                            if config["ALERT"]:
                                logger.info("Sending email alert..")
                                email_thread = threading.Thread(target=send_mail)
                                email_thread.daemon = True
                                email_thread.start()
                                logger.info("Alert sent!")
                        to.counted = True
                        total = []
                        total.append(len(move_in) - len(move_out))

            trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        info_status = [
            ("Exit", totalUp),
            ("Enter", totalDown),
            ("Status", status),
        ]

        info_threshold = [
            ("Total Inside", total_in - total_out),
        ]

        for (i, (k, v)) in enumerate(info_status):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for (i, (k, v)) in enumerate(info_threshold):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if writer is not None:
            writer.write(frame)

        if config["Log"]:
            log_data(move_in, in_time, move_out, out_time)

        fps.update()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        totalFrames += 1

    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))

    if writer is not None:
        writer.release()

    if not args.get("input", False):
        vs.stop()
    else:
        vs.release()

    cv2.destroyAllWindows()
    log_data(move_in, in_time, move_out, out_time) 
    
def get_counts():
    global total_in, total_out, total_inside
    return total_in, total_out, total_inside

# Static files configuration
app.config['STATIC_FOLDER'] = 'static'

# Blueprint for checkout functionality (if needed)
checkout_bp = Blueprint('checkout', __name__)

if __name__ == '__main__':
    app.secret_key = 'supersecretkey'
    
    # Start the people counter in a separate thread
    t = threading.Thread(target=run_people_counter)
    t.start()

    # Run the Flask app
    app.run(debug=True)

app.config['STATIC_FOLDER'] = 'static' 