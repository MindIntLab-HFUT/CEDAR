import os
import json
import time
import csv
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass
import base64
from PIL import Image
import io

REPO_ROOT = Path(__file__).resolve().parent.parent

# configuration
MODEL_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 128,
    "top_p": 1.0,
    "seed": 42,  
}

# language-specific prompts here
PROMPTS = {
    "arabic": """اقرأ الصورة والسؤال التاليين، واختر عاطفة واحدة تتطابق بشكل أفضل مع الموقف.
[السؤال]: {narrative}
{question}
[الخيارات]: غضب، اشمئزاز، خوف، سعادة، حزن، دهشة، تسلية، رهبة، قناعة، رغبة، إحراج، ألم، ارتياح، تعاطف
الآن، [إجابتك] هي:""",
    
    "chinese": """阅读以下图片和问题,选择一个最符合情境的情绪。
[问题]:{narrative}
{question}
[选项]:愤怒、厌恶、恐惧、幸福、悲伤、惊讶、愉悦、敬畏、满足、渴望、尴尬、痛苦、宽慰、同情
现在,[你的答案]是:""",
    
    "english": """Read the following image and question, and select ONE emotion that best matches the situation.
[question]: {narrative}
{question}
[options]: anger, disgust, fear, happiness, sadness, surprise, amusement, awe, contentment, desire, embarrassment, pain, relief, sympathy
Now, [your answer] is:""",
    
    "hindi": """निम्नलिखित छवि और प्रश्न को पढ़ें, और स्थिति से सबसे अच्छी तरह मेल खाने वाली एक भावना का चयन करें।
[प्रश्न]: {narrative}
{question}
[विकल्प]: गुस्सा, घृणा, डर, खुशी, उदासी, आश्चर्य, मज़ा, विस्मय, संतोष, इच्छा, शर्मिंदगी, दर्द, राहत, सहानुभूति
अब, [आपका उत्तर] है:""",
    
    "japanese": """以下の画像と質問を読み、状況に最もよく当てはまる感情を1つ選択してください。
[質問]: {narrative}
{question}
[選択肢]: 怒り, 嫌悪, 恐怖, 幸せ, 悲しみ, 驚き, 楽しみ, 畏敬, 満足, 欲望, 恥ずかしさ, 苦痛, 安堵, 同情
では、[あなたの答え]は:""",
    
    "spanish": """Lee la siguiente imagen y pregunta, y selecciona UNA emoción que mejor coincida con la situación.
[pregunta]: {narrative}
{question}
[opciones]: enojo, asco, miedo, felicidad, tristeza, sorpresa, diversión, asombro, contentamiento, deseo, vergüenza, dolor, alivio, compasión
Ahora, [tu respuesta] es:""",
    
    "swahili": """Soma picha na swali lifuatalo, na chagua hisia MOJA inayolingana vizuri zaidi na hali hiyo.
[swali]: {narrative}
{question}
[chaguo]: hasira, kinyaa, hofu, furaha, huzuni, mshangao, burudani, kicho, kuridhika, hamu, aibu, maumivu, afueni, huruma
Sasa, [jibu lako] ni:"""
}

# emotion mappings (each language -> English)
EMOTION_MAPPINGS = {
    "arabic": {
        "غضب": "anger", "اشمئزاز": "disgust", "خوف": "fear", "سعادة": "happiness",
        "حزن": "sadness", "دهشة": "surprise", "تسلية": "amusement", "رهبة": "awe",
        "قناعة": "contentment", "رغبة": "desire", "إحراج": "embarrassment", "ألم": "pain",
        "ارتياح": "relief", "تعاطف": "sympathy"
    },
    "chinese": {
        "愤怒": "anger", "厌恶": "disgust", "恐惧": "fear", "幸福": "happiness",
        "悲伤": "sadness", "惊讶": "surprise", "愉悦": "amusement", "敬畏": "awe",
        "满足": "contentment", "渴望": "desire", "尴尬": "embarrassment", "痛苦": "pain",
        "宽慰": "relief", "同情": "sympathy"
    },
    "english": {
        "anger": "anger", "disgust": "disgust", "fear": "fear", "happiness": "happiness",
        "sadness": "sadness", "surprise": "surprise", "amusement": "amusement", "awe": "awe",
        "contentment": "contentment", "desire": "desire", "embarrassment": "embarrassment",
        "pain": "pain", "relief": "relief", "sympathy": "sympathy"
    },
    "hindi": {
        "गुस्सा": "anger", "घृणा": "disgust", "डर": "fear", "खुशी": "happiness",
        "उदासी": "sadness", "आश्चर्य": "surprise", "मज़ा": "amusement", "विस्मय": "awe",
        "संतोष": "contentment", "इच्छा": "desire", "शर्मिंदगी": "embarrassment", "दर्द": "pain",
        "राहत": "relief", "सहानुभूति": "sympathy"
    },
    "japanese": {
        "怒り": "anger", "嫌悪": "disgust", "恐怖": "fear", "幸せ": "happiness",
        "悲しみ": "sadness", "驚き": "surprise", "楽しみ": "amusement", "畏敬": "awe",
        "満足": "contentment", "欲望": "desire", "恥ずかしさ": "embarrassment", "苦痛": "pain",
        "安堵": "relief", "同情": "sympathy"
    },
    "spanish": {
        "enojo": "anger", "asco": "disgust", "miedo": "fear", "felicidad": "happiness",
        "tristeza": "sadness", "sorpresa": "surprise", "diversión": "amusement", "asombro": "awe",
        "contentamiento": "contentment", "deseo": "desire", "vergüenza": "embarrassment",
        "dolor": "pain", "alivio": "relief", "compasión": "sympathy"
    },
    "swahili": {
        "hasira": "anger", "kinyaa": "disgust", "hofu": "fear", "furaha": "happiness",
        "huzuni": "sadness", "mshangao": "surprise", "burudani": "amusement", "kicho": "awe",
        "kuridhika": "contentment", "hamu": "desire", "aibu": "embarrassment", "maumivu": "pain",
        "afueni": "relief", "huruma": "sympathy"
    }
}

@dataclass
class EvaluationResult:
    """Data class for evaluation result"""
    id: int
    language: str
    narrative: str
    question: str
    ground_truth: str
    ground_truth_en: str
    model_response: str
    extracted_answer: str
    extracted_answer_en: str
    is_correct: Optional[bool]
    is_valid: bool
    error_message: str = ""


class ModelInterface:
    """Base class for model interface"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
    
    def generate(self, prompt: str, image_path: str) -> str:
        raise NotImplementedError


class VLLMModelInterface(ModelInterface):
    """VLLM local deployment interface"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, 
                 port: int = 8000, **kwargs):
        super().__init__(MODEL_CONFIG)
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.port = port
        self.base_url = f"http://localhost:{port}/v1"
        
        print(f"Please ensure VLLM server is running at {self.base_url}")
        print(f"Start command: python -m vllm.entrypoints.openai.api_server "
              f"--model {model_path} --tensor-parallel-size {tensor_parallel_size} "
              f"--port {port}")
    
    def generate(self, prompt: str, image_path: str) -> str:
        """generate response using VLLM API"""
        # encode image
        image_base64 = self._encode_image(image_path)
        
        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ],
            "temperature": self.model_config["temperature"],
            "max_tokens": self.model_config["max_tokens"],
            "top_p": self.model_config["top_p"],
            "seed": self.model_config.get("seed", 42)  
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"VLLM API call failed: {str(e)}")
    
    def _encode_image(self, image_path: str) -> str:
        """encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


class APIModelInterface(ModelInterface):
    """remote API interface"""
    
    def __init__(self, base_url: str, api_key: str, model_name: str, **kwargs):
        super().__init__(MODEL_CONFIG)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
    
    def generate(self, prompt: str, image_path: str) -> str:
        """generate response using remote API"""
        # encode image to base64
        image_base64 = self._encode_image(image_path)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ],
            "temperature": self.model_config["temperature"],
            "max_tokens": self.model_config["max_tokens"],
            "top_p": self.model_config["top_p"],
            "seed": self.model_config.get("seed", 42)  
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


class AnswerExtractor:
    """extract answers from model responses for different langs"""
    
    def __init__(self, language: str):
        self.language = language
        self.emotion_mapping = EMOTION_MAPPINGS[language]
    
    def extract(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        extract answer from model response, returns: (extracted_answer_in_language, extracted_answer_in_english)
        """
        if not response:
            return None, None
        
        # clean response
        response = response.strip()
        
        # try different extraction strategies
        extracted = None
        
        # s1: direct match with emotion keywords
        extracted = self._direct_match(response)
        
        # s2: extract first word
        if not extracted:
            extracted = self._extract_first_word(response)
        
        # s3: fuzzy match
        if not extracted:
            extracted = self._fuzzy_match(response)
        
        if extracted:
            # map to English
            extracted_en = self.emotion_mapping.get(extracted.lower(), None)
            if extracted_en:
                return extracted, extracted_en
        
        return None, None
    
    def _direct_match(self, response: str) -> Optional[str]:
        """direct match with emotion keywords"""
        response_lower = response.lower()
        
        for emotion_lang, emotion_en in self.emotion_mapping.items():
            if emotion_lang.lower() in response_lower:
                return emotion_lang
        
        return None
    
    def _extract_first_word(self, response: str) -> Optional[str]:
        """extract the first meaningful word"""
        # Remove common punctuation and get first word
        words = re.findall(r'\w+', response)
        
        if words:
            first_word = words[0]
            # Check if it matches any emotion
            for emotion in self.emotion_mapping.keys():
                if emotion.lower() == first_word.lower():
                    return emotion
        
        return None
    
    def _fuzzy_match(self, response: str) -> Optional[str]:
        """fuzzy match for partial matches"""
        response_lower = response.lower()
        
        # find emotions that are substrings of the response
        matches = []
        for emotion in self.emotion_mapping.keys():
            if emotion.lower() in response_lower or response_lower in emotion.lower():
                matches.append(emotion)
        
        if len(matches) == 1:
            return matches[0]
        
        # if multiple matches -> return the longest one
        if matches:
            return max(matches, key=len)
        
        return None


class EmotionEvaluator:
    """main evaluation class"""
    
    def __init__(self, json_folder: str, image_folder: str, 
                 model_interface: ModelInterface, output_dir: str = "./results"):
        self.json_folder = Path(json_folder)
        self.image_folder = Path(image_folder)
        self.model_interface = model_interface
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # get model name for checkpoint
        self.model_name = self._get_model_name()
        self.checkpoint_path = self.output_dir / f"{self.model_name}_checkpoint.json"
        
        # results storage
        self.all_results: List[EvaluationResult] = []
        self.processed_ids: set = set()
        
        # load checkpoint (if exists)
        self._load_checkpoint()
    
    def _get_model_name(self) -> str:
        """get model name from interface"""
        if isinstance(self.model_interface, VLLMModelInterface):
            return Path(self.model_interface.model_path).name
        else:
            return self.model_interface.model_name.replace('/', '_')
    
    def _load_checkpoint(self):
        """load checkpoint (if exists)"""
        if self.checkpoint_path.exists():
            print(f"loading checkpoint from: {self.checkpoint_path}")
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    
                for item in checkpoint:
                    result = EvaluationResult(**item)
                    self.all_results.append(result)
                    self.processed_ids.add(f"{result.language}_{result.id}")
                
                print(f"loaded {len(self.all_results)} previous results")
            except Exception as e:
                print(f"failed to load checkpoint: {e}")
    
    def _save_checkpoint(self):
        try:
            checkpoint = [vars(result) for result in self.all_results]
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"failed to save checkpoint: {e}")
    
    def evaluate(self):
        """run evaluation"""
        # find all jsons
        json_files = sorted(self.json_folder.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {self.json_folder}")
        
        print(f"Found {len(json_files)} JSON files")
        
        # process each json
        for json_file in json_files:
            language = self._extract_language(json_file.name)
            if language:
                print(f"\n{'='*60}")
                print(f"Processing {language} - {json_file.name}")
                print(f"{'='*60}")
                self._process_json_file(json_file, language)
        
        # save results
        self._save_checkpoint()
        self._save_results()
        
        # compute and save statistics
        self._compute_statistics()
        
        print("\n" + "="*60)
        print("Evaluation completed!")
        print(f"results saved to {self.output_dir}")
        print("="*60)
    
    def _extract_language(self, filename: str) -> Optional[str]:
        """extract language from filename"""
        for lang in PROMPTS.keys():
            if filename.lower().startswith(lang):
                return lang
        return None
    
    def _process_json_file(self, json_file: Path, language: str):
        """process a single json file"""
        # load data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        extractor = AnswerExtractor(language)
        prompt_template = PROMPTS[language]
        
        print(f"items: {len(data)}")
        
        for idx, item in enumerate(data):
            item_id = item['id']
            unique_id = f"{language}_{item_id}"
            
            # skip if already processed
            if unique_id in self.processed_ids:
                print(f"[{idx+1}/{len(data)}] skipping already processed item {item_id}")
                continue
            
            print(f"[{idx+1}/{len(data)}] processing item {item_id}...", end=" ")
            
            try:
                # find image
                image_path = self.image_folder / f"{item_id}.png"
                if not image_path.exists():
                    raise FileNotFoundError(f"image not found: {image_path}")
                
                # prepare prompt
                prompt = prompt_template.format(
                    narrative=item['narrative'],
                    question=item['question']
                )
                
                # call model
                response = self.model_interface.generate(prompt, str(image_path))
                
                # extract answer
                extracted_answer, extracted_answer_en = extractor.extract(response)
                
                # check correctness
                ground_truth = item['answer']
                ground_truth_en = item['english_ver_answer'].lower()
                
                is_valid = extracted_answer_en is not None
                is_correct = None
                
                if is_valid:
                    is_correct = (extracted_answer_en.lower() == ground_truth_en.lower())
                
                # store result
                result = EvaluationResult(
                    id=item_id,
                    language=language,
                    narrative=item['narrative'],
                    question=item['question'],
                    ground_truth=ground_truth,
                    ground_truth_en=ground_truth_en,
                    model_response=response,
                    extracted_answer=extracted_answer or "",
                    extracted_answer_en=extracted_answer_en or "",
                    is_correct=is_correct,
                    is_valid=is_valid
                )
                
                self.all_results.append(result)
                self.processed_ids.add(unique_id)
                
                status = "Correct" if is_correct else ("Wrong" if is_valid else "Invalid")
                print(status)
                
                # save checkpoint every 10 items
                if len(self.all_results) % 10 == 0:
                    self._save_checkpoint()
                
                # delay between requests
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                # store error result
                result = EvaluationResult(
                    id=item_id,
                    language=language,
                    narrative=item.get('narrative', ''),
                    question=item.get('question', ''),
                    ground_truth=item.get('answer', ''),
                    ground_truth_en=item.get('english_ver_answer', ''),
                    model_response="",
                    extracted_answer="",
                    extracted_answer_en="",
                    is_correct=None,
                    is_valid=False,
                    error_message=str(e)
                )
                self.all_results.append(result)
                self.processed_ids.add(unique_id)
                
                # save checkpoint on error
                self._save_checkpoint()
                
                # continue to next item
                continue
    
    def _save_results(self):
        """save detailed results to csv"""
        csv_path = self.output_dir / f"{self.model_name}_detailed_results.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'ID', 'Language', 'Narrative', 'Question', 
                'Ground Truth', 'Ground Truth (EN)', 
                'Model Response', 'Extracted Answer', 'Extracted Answer (EN)',
                'Is Correct', 'Is Valid', 'Error Message'
            ])
            
            for result in self.all_results:
                writer.writerow([
                    result.id, result.language, result.narrative, result.question,
                    result.ground_truth, result.ground_truth_en,
                    result.model_response, result.extracted_answer, result.extracted_answer_en,
                    result.is_correct, result.is_valid, result.error_message
                ])
        
        print(f"\ndetailed results saved to {csv_path}")
    
    def _compute_statistics(self):
        """compute and save statistics"""
        txt_path = self.output_dir / f"{self.model_name}_statistics.txt"
        
        # group results by language
        language_stats = {}
        for lang in PROMPTS.keys():
            lang_results = [r for r in self.all_results if r.language == lang]
            
            total = len(lang_results)
            valid = len([r for r in lang_results if r.is_valid])
            correct = len([r for r in lang_results if r.is_correct])
            invalid = total - valid
            
            accuracy = (correct / valid * 100) if valid > 0 else 0
            
            language_stats[lang] = {
                'total': total,
                'valid': valid,
                'correct': correct,
                'invalid': invalid,
                'accuracy': accuracy
            }
        
        # overall statistics
        total_all = len(self.all_results)
        valid_all = len([r for r in self.all_results if r.is_valid])
        correct_all = len([r for r in self.all_results if r.is_correct])
        invalid_all = total_all - valid_all
        accuracy_all = (correct_all / valid_all * 100) if valid_all > 0 else 0
        
        # write to file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"Evaluation Statistics for {self.model_name}\n")
            f.write("="*70 + "\n\n")
            
            f.write("Per-Language Results:\n")
            f.write("-"*70 + "\n")
            for lang, stats in sorted(language_stats.items()):
                f.write(f"\n{lang.upper()}:\n")
                f.write(f"  Total questions: {stats['total']}\n")
                f.write(f"  Valid responses: {stats['valid']}\n")
                f.write(f"  Correct answers: {stats['correct']}\n")
                f.write(f"  Invalid responses: {stats['invalid']}\n")
                f.write(f"  Accuracy: {stats['accuracy']:.2f}%\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Overall Results:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Total questions: {total_all}\n")
            f.write(f"  Valid responses: {valid_all}\n")
            f.write(f"  Correct answers: {correct_all}\n")
            f.write(f"  Invalid responses: {invalid_all}\n")
            f.write(f"  Overall Accuracy: {accuracy_all:.2f}%\n")
            f.write("="*70 + "\n")
            
            # configuration
            f.write("\nModel Configuration:\n")
            f.write("-"*70 + "\n")
            for key, value in MODEL_CONFIG.items():
                f.write(f"  {key}: {value}\n")
            f.write("="*70 + "\n")
        
        print(f"statistics saved to {txt_path}")
        
        # print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Overall Accuracy: {accuracy_all:.2f}%")
        print(f"Total: {total_all} | Valid: {valid_all} | Correct: {correct_all} | Invalid: {invalid_all}")
        print("="*70)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    MODEL_CONFIG['seed'] = seed


def main():
    """main function with example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal Tasks on Cedar")
    parser.add_argument("--json_folder", type=str,
                        default=str(REPO_ROOT / "data" / "multimodal" / "multilingual"),
                        help="path to folder containing {language}_*.json files")
    parser.add_argument("--image_folder", type=str,
                        default=str(REPO_ROOT / "data" / "multimodal" / "images"),
                        help="path to folder containing images")
    parser.add_argument("--output_dir", type=str, default="./results/multimodal", help="output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility") 
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0.0 = greedy decoding)")
    parser.add_argument("--max_tokens", type=int, default=128, help="max new tokens per response")
    
    # model selection
    parser.add_argument("--mode", type=str, choices=["vllm", "api"], required=True, 
                       help="model deployment mode: vllm or api")
    
    # if mode: vllm -> vLLM parameters
    parser.add_argument("--model_path", type=str, help="path to local model (for vllm mode)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                       help="tensor parallel size for vllm")
    parser.add_argument("--port", type=int, default=8000, help="port for vllm server")
    
    # if mode: api -> api parameters
    parser.add_argument("--base_url", type=str, help="base url for api (for api mode)")
    parser.add_argument("--api_key", type=str, help="api key (for api mode)")
    parser.add_argument("--model_name", type=str, help="model name (for api mode)")
    
    args = parser.parse_args()
    
    MODEL_CONFIG["temperature"] = args.temperature
    MODEL_CONFIG["max_tokens"] = args.max_tokens
    set_seed(args.seed)
    
    '''
    Example usage
    For vLLM (start `vllm serve /path/to/your/model --port 8000` first):
    python evaluation/run_benchmark_multimodal.py --mode vllm --model_path /path/to/your/model --port 8000

    For an OpenAI-compatible API:
    python evaluation/run_benchmark_multimodal.py --mode api \
        --base_url https://api.openai.com/v1 --api_key sk-xxx --model_name your_model_name
    '''
    
    # initialize model interface
    if args.mode == "vllm":
        if not args.model_path:
            raise ValueError("--model_path is required for vllm mode")
        model_interface = VLLMModelInterface(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            port=args.port
        )
    else:  # api mode
        if not all([args.base_url, args.api_key, args.model_name]):
            raise ValueError("--base_url, --api_key, and --model_name are required for api mode")
        model_interface = APIModelInterface(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name
        )
    
    # create evaluator and run
    evaluator = EmotionEvaluator(
        json_folder=args.json_folder,
        image_folder=args.image_folder,
        model_interface=model_interface,
        output_dir=args.output_dir
    )
    
    evaluator.evaluate()


if __name__ == "__main__":
    main()