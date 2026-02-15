"""
LLM Service - Medical Report Generation and Chat using Google Gemini API
Replaces local MedAlpaca model with cloud-based Gemini for improved performance
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG

# Load environment variables
load_dotenv()


class LLMService:
    def __init__(self):
        self.model = None
        self.config = MODEL_CONFIG["llm"]
        self.api_key = None
        
    def load_model(self):
        """Initialize Gemini API client"""
        try:
            print("[LLM] Initializing Gemini API...")
            
            # Get API key from environment
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY not found in environment variables. "
                    "Please create a .env file with your API key."
                )
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Initialize model
            model_name = self.config["model_name"]
            print(f"[LLM] Using model: {model_name}")
            
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": self.config["temperature"],
                    "top_p": self.config["top_p"],
                    "max_output_tokens": self.config["max_tokens"],
                }
            )
            
            print(f"[LLM] Gemini API initialized successfully")
            return True
            
        except Exception as e:
            print(f"[LLM] Error initializing Gemini API: {str(e)}")
            raise
    
    def construct_report_prompt(self, findings: list, rag_context: str) -> str:
        """
        Construct a medical prompt for report generation
        
        Args:
            findings: List of detected pathologies
            rag_context: Similar case reports from RAG
            
        Returns:
            Formatted prompt string
        """
        findings_str = ", ".join([f"{f['name']} ({f['confidence']:.2f})" for f in findings])
        
        prompt = f"""You are an expert radiologist writing a professional chest X-ray report.

**Detected Findings:**
{findings_str}

**Similar Historical Cases for Reference:**
{rag_context}

**Task:** Generate a structured radiology report with the following sections:

1. **FINDINGS**: Describe the observed pathologies in detail based on the detected findings
2. **IMPRESSION**: Provide clinical interpretation and significance
3. **RECOMMENDATIONS**: Suggest follow-up actions if needed

Write in professional medical language. Be concise but thorough. Use markdown formatting for sections.

**REPORT:**
"""
        return prompt
    
    def construct_chat_prompt(self, history: list, user_input: str, case_context: str = "") -> str:
        """
        Construct a chat prompt with conversation history
        
        Args:
            history: List of previous messages
            user_input: Current user question
            case_context: Current case information
            
        Returns:
            Formatted prompt string
        """
        system_context = """You are a medical AI assistant helping doctors understand radiology reports. 
Answer questions clearly and professionally based on medical knowledge and the current case."""
        
        if case_context:
            system_context += f"\n\n**Current Case:**\n{case_context}"
        
        # Build conversation history
        conversation = f"{system_context}\n\n"
        for msg in history[-6:]:  # Keep last 6 messages
            content = msg.get("content", "")
            conversation += f"{content}\n\n"
        
        conversation += f"{user_input}\n\nAssistant:"
        
        return conversation
    
    def generate_report(self, findings: list, rag_context: str) -> str:
        """
        Generate a structured medical report using Gemini API
        
        Args:
            findings: Pathologies detected by vision model
            rag_context: Similar cases from RAG
            
        Returns:
            Generated report text
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Construct prompt
            prompt = self.construct_report_prompt(findings, rag_context)
            
            # Generate with Gemini
            print("[LLM] Generating report with Gemini...")
            response = self.model.generate_content(prompt)
            
            # Extract text from response
            if response and response.text:
                report = response.text.strip()
                
                # Remove "**REPORT:**" prefix if present
                if "**REPORT:**" in report:
                    report = report.split("**REPORT:**")[1].strip()
                
                return report
            else:
                return "Unable to generate report. Please try again."
            
        except Exception as e:
            print(f"[LLM] Report generation error: {str(e)}")
            # Return a fallback report
            findings_str = ", ".join([f"{f['name']}" for f in findings])
            return f"""**FINDINGS:**
The chest X-ray shows evidence of {findings_str}.

**IMPRESSION:**
Clinical correlation is recommended.

**RECOMMENDATIONS:**
Follow-up imaging may be considered based on clinical presentation.

*Note: Automated report generation encountered an error. Please review findings manually.*
"""
    
    def chat(self, history: list, user_input: str, case_context: str = "") -> str:
        """
        Handle conversational queries about the case using Gemini API
        
        Args:
            history: Previous messages
            user_input: User's question
            case_context: Current case information
            
        Returns:
            Assistant's response
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Construct prompt
            prompt = self.construct_chat_prompt(history, user_input, case_context)
            
            # Generate with Gemini
            print("[LLM] Generating chat response with Gemini...")
            response = self.model.generate_content(prompt)
            
            # Extract text from response
            if response and response.text:
                return response.text.strip()
            else:
                return "I apologize, but I'm having trouble generating a response. Please try again."
            
        except Exception as e:
            print(f"[LLM] Chat error: {str(e)}")
            return "I encountered an error processing your question. Please try rephrasing or try again."


# Singleton instance
llm_service = LLMService()


if __name__ == "__main__":
    # Test the service
    print("Testing Gemini LLM Service...")
    llm_service.load_model()
    
    # Test report generation
    test_findings = [
        {"name": "Cardiomegaly", "confidence": 0.85},
        {"name": "Edema", "confidence": 0.72}
    ]
    test_context = "Previous case showed similar cardiomegaly with pulmonary congestion."
    
    report = llm_service.generate_report(test_findings, test_context)
    print("\n" + "="*50)
    print("GENERATED REPORT:")
    print("="*50)
    print(report)
