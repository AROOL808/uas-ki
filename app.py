import os
import cv2
import numpy as np
import itertools
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import rsa
import requests
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# DCT Class (from your reference)
quant = np.array([[16,11,10,16,24,40,51,61],
                [12,12,14,19,26,58,60,55],
                [14,13,16,24,40,57,69,56],
                [14,17,22,29,51,87,80,62],
                [18,22,37,56,68,109,103,77],
                [24,35,55,64,81,104,113,92],
                [49,64,78,87,103,121,120,101],
                [72,92,95,98,112,100,103,99]])

class DCT():    
    def __init__(self):
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0   
    
    def encode_image(self, img, secret_msg):
        secret = secret_msg
        self.message = str(len(secret))+'*'+secret
        self.bitMess = self.toBits()
        
        row, col = img.shape[:2]
        self.oriRow, self.oriCol = row, col  
        
        if ((col/8)*(row/8) < len(secret)):
            raise ValueError("Error: Message too large to encode in image")
        
        if row%8 != 0 or col%8 != 0:
            img = self.addPadd(img, row, col)
        
        row, col = img.shape[:2]
        bImg, gImg, rImg = cv2.split(img)
        bImg = np.float32(bImg)
        
        imgBlocks = [np.round(bImg[j:j+8, i:i+8]-128) for (j,i) in itertools.product(range(0,row,8), range(0,col,8))]
        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
        quantizedDCT = [np.round(dct_Block/quant) for dct_Block in dctBlocks]
        
        messIndex = 0
        letterIndex = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            DC[7] = self.bitMess[messIndex][letterIndex]
            DC = np.packbits(DC)
            DC = np.float32(DC)
            DC = DC-255
            quantizedBlock[0][0] = DC
            letterIndex += 1
            if letterIndex == 8:
                letterIndex = 0
                messIndex += 1
                if messIndex == len(self.message):
                    break
        
        sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
        sImg = []
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg, gImg, rImg))
        return sImg

    def decode_image(self, img):
        row, col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0
        
        bImg, gImg, rImg = cv2.split(img)
        bImg = np.float32(bImg)
        
        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j,i) in itertools.product(range(0,row,8), range(0,col,8))]    
        quantizedDCT = [img_Block/quant for img_Block in imgBlocks]
        
        i = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            if DC[7] == 1:
                buff += (0 & 1) << (7-i)
            elif DC[7] == 0:
                buff += (1&1) << (7-i)
            i += 1
            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i = 0
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                    except:
                        pass
            if messSize and len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize))+1:]
        return ''
      
    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]
            
    def addPadd(self, img, row, col):
        img = cv2.resize(img, (col+(8-col%8), row+(8-row%8)))    
        return img
        
    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8,'0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8,'0')
        return bits

# Helper functions
def generate_rsa_keys():
    (pubkey, privkey) = rsa.newkeys(512)
    return pubkey, privkey

def save_key_to_file(key, filename):
    with open(filename, 'wb') as f:
        f.write(key.save_pkcs1())

def load_key_from_file(filename, key_type='public'):
    with open(filename, 'rb') as f:
        if key_type == 'public':
            return rsa.PublicKey.load_pkcs1(f.read())
        else:
            return rsa.PrivateKey.load_pkcs1(f.read())

def generate_ai_image(prompt):
    # Replace with your actual API key
    api_key = 'cc25e7abe4af8aa1d6d1af86691321cb5393a2536598f0286b0fce6b48e66424195353ecca69a18c11e647b72df07ae7'
    
    response = requests.post(
        'https://clipdrop-api.co/text-to-image/v1',
        files={'prompt': (None, prompt, 'text/plain')},
        headers={'x-api-key': api_key}
    )
    
    if response.ok:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception(f"AI Image generation failed: {response.status_code}")

def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['GET', 'POST'])
def encode():
    if request.method == 'POST':
        try:
            # Get form data
            prompt = request.form['prompt']
            message = request.form['message']
            public_key_text = request.form['public_key']
            
            # Generate AI image
            ai_image = generate_ai_image(prompt)
            
            # Convert PIL Image to OpenCV format
            img_cv = cv2.cvtColor(np.array(ai_image), cv2.COLOR_RGB2BGR)
            
            # Encrypt the message with RSA
            try:
                pub_key = rsa.PublicKey.load_pkcs1(public_key_text.encode())
            except:
                return render_template('encode.html', error="Invalid public key format")
            
            encrypted_msg = rsa.encrypt(message.encode(), pub_key)
            encrypted_msg_base64 = base64.b64encode(encrypted_msg).decode()
            
            # Encode message in image using DCT
            dct = DCT()
            encoded_img = dct.encode_image(img_cv, encrypted_msg_base64)
            
            # Convert back to PIL Image
            encoded_img_rgb = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB)
            encoded_pil_img = Image.fromarray(encoded_img_rgb)
            
            # Save the encoded image
            filename = secure_filename('encoded_image.png')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            encoded_pil_img.save(filepath)
            
            # Prepare image for display
            img_io = io.BytesIO()
            encoded_pil_img.save(img_io, 'PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode()
            
            return render_template('encode_result.html', 
                                 image_data=img_base64,
                                 filename=filename)
            
        except Exception as e:
            return render_template('encode.html', error=str(e))
    
    return render_template('encode.html')

@app.route('/decode', methods=['GET', 'POST'])
def decode():
    if request.method == 'POST':
        try:
            # Get form data
            private_key_text = request.form['private_key']
            
            # Check if the post request has the file part
            if 'image' not in request.files:
                return render_template('decode.html', error="No image file uploaded")
            
            file = request.files['image']
            if file.filename == '':
                return render_template('decode.html', error="No image selected")
            
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read the image with OpenCV
            img_cv = cv2.imread(filepath)
            if img_cv is None:
                return render_template('decode.html', error="Invalid image file")
            
            # Decode the message from the image
            dct = DCT()
            encrypted_msg_base64 = dct.decode_image(img_cv)
            
            if not encrypted_msg_base64:
                return render_template('decode.html', error="No message found in image")
            
            # Decrypt the message with RSA
            try:
                priv_key = rsa.PrivateKey.load_pkcs1(private_key_text.encode())
            except:
                return render_template('decode.html', error="Invalid private key format")
            
            try:
                encrypted_msg = base64.b64decode(encrypted_msg_base64)
                decrypted_msg = rsa.decrypt(encrypted_msg, priv_key).decode()
            except:
                return render_template('decode.html', error="Decryption failed - wrong key?")
            
            # Prepare image for display
            img_pil = Image.open(filepath)
            img_io = io.BytesIO()
            img_pil.save(img_io, 'PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode()
            
            return render_template('decode_result.html', 
                                 image_data=img_base64,
                                 message=decrypted_msg)
            
        except Exception as e:
            return render_template('decode.html', error=str(e))
    
    return render_template('decode.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/generate_keys', methods=['POST'])
def generate_keys():
    pubkey, privkey = generate_rsa_keys()
    # Create a response with the private key as a downloadable file
    response = {
        'public_key': pubkey.save_pkcs1().decode(),
        'private_key': privkey.save_pkcs1().decode()
    }
    return response

if __name__ == '__main__':
    app.run(debug=True)