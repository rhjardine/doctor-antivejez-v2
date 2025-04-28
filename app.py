from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import json
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from the frontend

# Simulated storage directory for uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Simulated knowledge base
KNOWLEDGE_BASE = {
    'labTests': {
        'Glucose': {'normalRange': '70-99 mg/dL', 'high': 'May indicate prediabetes or diabetes', 'low': 'May indicate hypoglycemia'},
        'HbA1c': {'normalRange': '<5.7%', 'high': 'Indicates poor blood sugar control', 'low': 'Uncommon, may indicate hypoglycemia'},
        'Homocysteine': {'normalRange': '<10 μmol/L', 'high': 'May indicate methylation issues', 'low': 'Typically not a concern'},
        'hsCRP': {'normalRange': '<1.0 mg/L', 'high': 'Indicates systemic inflammation', 'low': 'Typically not a concern'}
    },
    'geneticVariants': {
        'TCF7L2 rs7903146': {'CT': '40% increased risk of type 2 diabetes', 'TT': 'Higher risk of type 2 diabetes'},
        'MTHFR C677T': {'CT': 'Reduced enzyme activity (~30%), affects folate metabolism', 'TT': 'Significantly reduced enzyme activity, higher risk of elevated homocysteine'}
    },
    'epigeneticMarkers': {
        'DNA Methylation Age': {'high': 'Accelerated aging', 'low': 'Decelerated aging'}
    }
}

# Simulated patient data storage
PATIENT_DATA = {
    '458912': {  # Isabel Romero's ID
        'name': 'Isabel Romero',
        'age': 58,
        'gender': 'Female',
        'biologicalAge': 52.3,
        'healthScore': 84,
        'lastCheckup': '2023-04-12',
        'documents': []
    }
}

@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient_data(patient_id):
    patient = PATIENT_DATA.get(patient_id)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    return jsonify(patient)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the file
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Simulate file metadata
    file_metadata = {
        'id': file_id,
        'name': file.filename,
        'type': get_file_type(file),
        'size': os.path.getsize(file_path),
        'uploadDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add to patient documents
    PATIENT_DATA['458912']['documents'].append(file_metadata)
    
    return jsonify({
        'message': 'File uploaded successfully',
        'file': file_metadata
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    data = request.json
    if not data or 'fileId' not in data:
        return jsonify({'error': 'File ID required'}), 400
    
    file_id = data['fileId']
    document = next((doc for doc in PATIENT_DATA['458912']['documents'] if doc['id'] == file_id), None)
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    
    analysis = perform_document_analysis(document)
    document['analysis'] = analysis
    
    return jsonify({
        'message': 'Document analyzed successfully',
        'analysis': analysis
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Message required'}), 400
    
    message = data['message']
    context = data.get('context', {})
    
    response = generate_chat_response(message, context)
    return jsonify({
        'response': response
    })

def get_file_type(file):
    extension = file.filename.split('.')[-1].lower()
    if file.mimetype == 'application/pdf':
        return 'PDF Document'
    if file.mimetype.startswith('image/'):
        return 'Image'
    if file.mimetype in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        return 'Word Document'
    if file.mimetype in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] or extension == 'csv':
        return 'Spreadsheet'
    if file.mimetype == 'text/plain':
        return 'Text Document'
    if 'genomic' in file.filename.lower() or 'genetic' in file.filename.lower() or 'dna' in file.filename.lower():
        return 'Genomic Data'
    if 'epigenetic' in file.filename.lower():
        return 'Epigenetic Report'
    return 'Document'

def perform_document_analysis(document):
    filename = document['name'].lower()
    analysis = {'type': document['type'], 'results': []}
    
    if 'genomic' in filename or 'genetic' in filename or 'dna' in filename:
        analysis['results'] = [
            {'gene': 'TCF7L2', 'variant': 'rs7903146', 'genotype': 'CT', 'interpretation': KNOWLEDGE_BASE['geneticVariants']['TCF7L2 rs7903146']['CT']},
            {'gene': 'MTHFR', 'variant': 'C677T', 'genotype': 'CT', 'interpretation': KNOWLEDGE_BASE['geneticVariants']['MTHFR C677T']['CT']}
        ]
        analysis['recommendations'] = ['Consider time-restricted eating for TCF7L2', 'Methylated B vitamins for MTHFR']
    elif 'blood' in filename or 'lab' in filename:
        analysis['results'] = [
            {'name': 'Glucose', 'value': '105 mg/dL', 'range': KNOWLEDGE_BASE['labTests']['Glucose']['normalRange'], 'interpretation': KNOWLEDGE_BASE['labTests']['Glucose']['high']},
            {'name': 'HbA1c', 'value': '5.6%', 'range': KNOWLEDGE_BASE['labTests']['HbA1c']['normalRange'], 'interpretation': 'Within normal range'},
            {'name': 'Homocysteine', 'value': '12.4 μmol/L', 'range': KNOWLEDGE_BASE['labTests']['Homocysteine']['normalRange'], 'interpretation': KNOWLEDGE_BASE['labTests']['Homocysteine']['high']}
        ]
        analysis['recommendations'] = ['Monitor for insulin resistance due to elevated glucose']
    elif 'epigenetic' in filename:
        analysis['results'] = [
            {'name': 'DNA Methylation Age', 'status': 'Slightly elevated', 'interpretation': KNOWLEDGE_BASE['epigeneticMarkers']['DNA Methylation Age']['high']}
        ]
        analysis['recommendations'] = ['Interventions to reduce oxidative stress']
    else:
        analysis['results'] = [{'name': 'General Analysis', 'interpretation': 'Please specify document type for detailed analysis'}]
    
    return analysis

def generate_chat_response(message, context):
    msg_lower = message.lower()
    if context.get('analysis'):
        if context['analysis']['type'] == 'lab' and 'normal range' in msg_lower:
            return '\n'.join([f"{test['name']}: {test['range']}" for test in context['analysis']['results']])
        if context['analysis']['type'] == 'genetic' and 'more about' in msg_lower:
            for result in context['analysis']['results']:
                if result['gene'].lower() in msg_lower:
                    return f"{result['gene']} ({result['variant']}): Genotype {result['genotype']} - {result['interpretation']}"
    if 'interpret' in msg_lower or 'analyze' in msg_lower:
        return 'Please upload a document for detailed analysis and interpretation.'
    return 'I’m here to assist with Isabel’s health. You can upload medical reports, genetic data, lab results, or epigenetic reports for analysis.'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)