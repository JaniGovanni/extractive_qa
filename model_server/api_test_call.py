import requests
import json

# API endpoint
url = "http://localhost:8503/api/v1/question_answering"

# Sample input data
data = {
    "question": "What are the main challenges faced by the Amazon rainforest and what solutions have been proposed?",
    "context": """
    The Amazon rainforest, spanning across nine countries in South America, is the world's largest tropical rainforest and is often referred to as the "lungs of the Earth." It plays a crucial role in global climate regulation and biodiversity conservation. However, the Amazon faces numerous challenges:

    1. Deforestation: Large-scale clearing of forest for agriculture, cattle ranching, and logging has led to significant loss of forest cover. Between 1978 and 2020, over 750,000 square kilometers of forest were lost.

    2. Climate change: Rising temperatures and changing precipitation patterns are affecting the forest's ecosystem, potentially leading to more frequent droughts and fires.

    3. Biodiversity loss: As habitat is destroyed, countless species of plants and animals are threatened with extinction.

    4. Indigenous rights: The rights and livelihoods of indigenous communities living in the Amazon are often overlooked in development plans.

    To address these challenges, several solutions have been proposed:

    1. Stricter enforcement of anti-deforestation laws and policies.
    2. Promotion of sustainable agriculture and forestry practices.
    3. Creation of more protected areas and indigenous reserves.
    4. Investment in alternative economic activities that don't rely on deforestation.
    5. International cooperation and funding for conservation efforts.
    6. Implementation of REDD+ (Reducing Emissions from Deforestation and Forest Degradation) programs.
    7. Increased use of satellite monitoring and other technologies to track forest cover.

    Despite these efforts, the Amazon continues to face significant threats, and ongoing commitment from governments, NGOs, and the international community is crucial for its preservation.
    """
    }

# Send POST request
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    
    # Print the result
    print("API Response:")
    print(json.dumps(result, indent=2))
else:
    print(f"Error: API request failed with status code {response.status_code}")
    print(response.text)