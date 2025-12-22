try:
    import google.generativeai as genai
    print("google.generativeai is available")
except ImportError:
    print("google.generativeai is NOT available")

try:
    from google import genai
    print("google.genai (new sdk) is available")
except ImportError:
    print("google.genai (new sdk) is NOT available")
