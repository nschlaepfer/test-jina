INPUT_SCHEMA = {
    "query": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["What is Inferless?"]
    },
    "chunks": {
        'datatype': 'STRING',
        'required': True,
        'shape': [-1],
        'example': [
            "Inferless is a machine learning model deployment platform.",
            "It simplifies the deployment process for machine learning models.",
            "Users can deploy models without worrying about infrastructure."
        ]
    }
}
