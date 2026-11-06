import os
import re
import csv
import json
import time
import logging
import argparse
import random 

from tqdm import tqdm 
from pathlib import Path
from openai import OpenAI
try:
    from vllm import LLM, SamplingParams
except ImportError:  # vllm is only required for --mode vllm
    LLM, SamplingParams = None, None
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any


# configs
GENERATION_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 128,  
    "top_p": 1.0
}

# avoid rate limits
REQUEST_INTERVAL = 0.01

# log (a file handler is added in main() once the output dir is known)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    GENERATION_CONFIG["seed"] = seed
    logger.info(f"Set global random seed to {seed}")

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class EvaluationItem:

    language: str
    index: int
    narrative: str
    question: str
    answer: str
    english_ver_answer: str
    model_response: str = ""
    extracted_answer: str = ""
    is_correct: bool = False
    is_valid: bool = True
    

@dataclass
class LanguageResults:

    language: str
    total: int = 0
    correct: int = 0
    invalid: int = 0
    accuracy: float = 0.0


# prompt templates for each language
PROMPT_TEMPLATES = {
    "arabic": """اقرأ السرد التالي واختر عاطفة واحدة تتطابق بشكل أفضل مع الموقف.
[السرد]: {narrative}
[السؤال]: {question}
[الخيارات]: غضب، اشمئزاز، خوف، سعادة، حزن، دهشة، تسلية، رهبة، قناعة، رغبة، إحراج، ألم، ارتياح، تعاطف
الآن، [إجابتك] هي:""",
    
    "chinese": """阅读以下叙述，并选择一种最符合情境的情绪。
[叙述]：{narrative}
[问题]：{question}
[选项]：愤怒、厌恶、恐惧、幸福、悲伤、惊讶、愉悦、敬畏、满足、渴望、尴尬、痛苦、宽慰、同情
现在，[你的答案]是：""",
    
    "english": """Read the following narrative and select ONE emotion that best matches the situation.
[narrative]: {narrative}
[question]: {question}
[options]: anger, disgust, fear, happiness, sadness, surprise, amusement, awe, contentment, desire, embarrassment, pain, relief, sympathy
Now, [your answer] is:""",
    
    "hindi": """निम्नलिखित कथन को पढ़ें और उस स्थिति से सबसे अच्छी तरह मेल खाने वाली एक भावना चुनें।
[कथन]: {narrative}
[प्रश्न]: {question}
[विकल्प]: गुस्सा, घृणा, डर, खुशी, उदासी, आश्चर्य, मज़ा, विस्मय, संतोष, इच्छा, शर्मिंदगी, दर्द, राहत, सहानुभूति
अब, [आपका उत्तर] है:""",
    
    "japanese": """以下の物語を読んで、その状況に最も適した感情を1つ選んでください。
[物語]: {narrative}
[質問]: {question}
[選択肢]: 怒り, 嫌悪, 恐怖, 幸せ, 悲しみ, 驚き, 楽しみ, 畏敬, 満足, 欲望, 恥ずかしさ, 苦痛, 安堵, 同情
それでは、[あなたの答え]は:""",
    
    "spanish": """Lee la siguiente narrativa y selecciona UNA emoción que mejor corresponda a la situación.
[narrativa]: {narrative}
[pregunta]: {question}
[opciones]: enojo, asco, miedo, felicidad, tristeza, sorpresa, diversión, asombro, contentamiento, deseo, vergüenza, dolor, alivio, compasión
Ahora, [tu respuesta] es:""",
    
    "swahili": """Soma simulizi ifuatayo na chagua hisia MOJA inayolingana zaidi na hali hiyo.
[simulizi]: {narrative}
[swali]: {question}
[chaguo]: hasira, kinyaa, hofu, furaha, huzuni, mshangao, burudani, kicho, kuridhika, hamu, aibu, maumivu, afueni, huruma
Sasa, [jibu lako] ni:"""
}


# emotion mappings for each language
EMOTION_MAPPINGS = {
    "arabic": {
        "غضب": "anger",
        "اشمئزاز": "disgust",
        "خوف": "fear",
        "سعادة": "happiness",
        "حزن": "sadness",
        "دهشة": "surprise",
        "تسلية": "amusement",
        "رهبة": "awe",
        "قناعة": "contentment",
        "رغبة": "desire",
        "إحراج": "embarrassment",
        "ألم": "pain",
        "ارتياح": "relief",
        "تعاطف": "sympathy"
    },
    "chinese": {
        "愤怒": "anger",
        "厌恶": "disgust",
        "恐惧": "fear",
        "幸福": "happiness",
        "悲伤": "sadness",
        "惊讶": "surprise",
        "愉悦": "amusement",
        "敬畏": "awe",
        "满足": "contentment",
        "渴望": "desire",
        "尴尬": "embarrassment",
        "痛苦": "pain",
        "宽慰": "relief",
        "同情": "sympathy"
    },
    "english": {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "happiness": "happiness",
        "sadness": "sadness",
        "surprise": "surprise",
        "amusement": "amusement",
        "awe": "awe",
        "contentment": "contentment",
        "desire": "desire",
        "embarrassment": "embarrassment",
        "pain": "pain",
        "relief": "relief",
        "sympathy": "sympathy"
    },
    "hindi": {
        "गुस्सा": "anger",
        "घृणा": "disgust",
        "डर": "fear",
        "खुशी": "happiness",
        "उदासी": "sadness",
        "आश्चर्य": "surprise",
        "मज़ा": "amusement",
        "विस्मय": "awe",
        "संतोष": "contentment",
        "इच्छा": "desire",
        "शर्मिंदगी": "embarrassment",
        "दर्द": "pain",
        "राहत": "relief",
        "सहानुभूति": "sympathy"
    },
    "japanese": {
        "怒り": "anger",
        "嫌悪": "disgust",
        "恐怖": "fear",
        "幸せ": "happiness",
        "悲しみ": "sadness",
        "驚き": "surprise",
        "楽しみ": "amusement",
        "畏敬": "awe",
        "満足": "contentment",
        "欲望": "desire",
        "恥ずかしさ": "embarrassment",
        "苦痛": "pain",
        "安堵": "relief",
        "同情": "sympathy"
    },
    "spanish": {
        "enojo": "anger",
        "asco": "disgust",
        "miedo": "fear",
        "felicidad": "happiness",
        "tristeza": "sadness",
        "sorpresa": "surprise",
        "diversión": "amusement",
        "asombro": "awe",
        "contentamiento": "contentment",
        "deseo": "desire",
        "vergüenza": "embarrassment",
        "dolor": "pain",
        "alivio": "relief",
        "compasión": "sympathy"
    },
    "swahili": {
        "hasira": "anger",
        "kinyaa": "disgust",
        "hofu": "fear",
        "furaha": "happiness",
        "huzuni": "sadness",
        "mshangao": "surprise",
        "burudani": "amusement",
        "kicho": "awe",
        "kuridhika": "contentment",
        "hamu": "desire",
        "aibu": "embarrassment",
        "maumivu": "pain",
        "afueni": "relief",
        "huruma": "sympathy"
    }
}


# extract answer from model response
class AnswerExtractor:
    
    @staticmethod
    def extract_answer(response: str, language: str) -> Optional[str]:
    # response, language -> return extracted answer or None

        if not response:
            return None
            
        response = response.strip().lower()
        emotion_map = EMOTION_MAPPINGS.get(language, {})
        
        # 1, map words directly
        for native_emotion, english_emotion in emotion_map.items():
            if native_emotion.lower() in response:
                logger.debug(f"Map words directly: {native_emotion} -> {english_emotion}")
                return english_emotion
        
        # 2, map english words directly
        for english_emotion in EMOTION_MAPPINGS["english"].keys():
            if english_emotion in response:
                logger.debug(f"Map english words directly: {english_emotion}")
                return english_emotion
        
        # 3, pattern matching
        if language == "chinese":
            # for chinese
            patterns = [
                r'答案[是为：:]\s*["""]?([^"""\s，。]+)',
                r'选择[是为：:]\s*[""""]?([^"""\s，。]+)',
                r'情绪[是为：:]\s*["""]?([^"""\s，。]+)',
                r'[是为：:]\s*["""]?([^"""\s，。]+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    extracted = match.group(1).strip()
                    for native, english in emotion_map.items():
                        if native in extracted:
                            return english
        
        elif language == "english":
            # for english
            patterns = [
                r'answer\s+is[:\s]+([a-z]+)',
                r'emotion\s+is[:\s]+([a-z]+)',
                r'is[:\s]+([a-z]+)',
                r'^([a-z]+)$',
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    extracted = match.group(1).strip()
                    if extracted in emotion_map:
                        return extracted
        
        elif language == "japanese":
            # for japanese
            patterns = [
                r'答え[はが：:]\s*["""]?([^"""\s、。]+)',
                r'[はが：:]\s*["""]?([^"""\s、。]+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    extracted = match.group(1).strip()
                    for native, english in emotion_map.items():
                        if native in extracted:
                            return english
        
        elif language == "arabic":
            # for arabic
            for native, english in emotion_map.items():
                if native in response:
                    return english
        
        # 4, fuzzy matching
        words = re.split(r'[\s,，.。:：;；!！?？\'"""]+', response)
        for word in words:
            word = word.strip()
            if word in emotion_map:
                logger.debug(f"Fuzzy matching to {word}")
                return emotion_map[word]
            # check
            if word in EMOTION_MAPPINGS["english"]:
                return word
        
        logger.warning(f"Failed to extract answer from model response: {response[:100]}")
        return None


# model interfaces
class ModelInterface:
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = None
        
    def generate(self, prompt: str) -> str:
    # generate model response
        raise NotImplementedError
        
    def close(self):
        pass


class VLLMModel(ModelInterface):
    
    def __init__(self, model_path: str, model_name: str, 
                 tensor_parallel_size: int = 1):
        super().__init__(model_name)
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        
        if LLM is None or SamplingParams is None:
            logger.error("Failed to import vllm library")
            raise ImportError("vllm library is not installed")

        # initialize vllm
        logger.info(f"Loading vllm model from {self.model_path}")
        logger.info(f"Tensor parallel size: {self.tensor_parallel_size}")

        try:
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                seed=GENERATION_CONFIG.get("seed", 42)
            )
        except Exception as e:
            logger.error(f"Failed to import vllm: {e}")
            raise

        self.sampling_params = SamplingParams(**GENERATION_CONFIG)
        logger.info("vllm model loaded successfully")
        
    def generate(self, prompt: str) -> str:

        try:

            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            
            # extract text
            if outputs and outputs[0].outputs:
                return outputs[0].outputs[0].text
            else:
                logger.warning("empty response")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to generate responses: {e}")
            return ""
    
    def close(self):
        pass


class APIModel(ModelInterface):
    
    def __init__(self, base_url: str, api_key: str, model_name: str):
        super().__init__(model_name)
        self.base_url = base_url
        self.api_key = api_key
        
        # openai client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        logger.info(f"Initialize api model: {model_name}")
        logger.info(f"Base url: {base_url}")
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **GENERATION_CONFIG
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Api error: {e}")
            return ""


class Evaluator:
    
    def __init__(self, data_dir: str, model: ModelInterface, output_dir: str = "results"):
        self.data_dir = Path(data_dir)
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_output_dir = self.output_dir / self._sanitize_model_name(model.model_name)
        self.model_output_dir.mkdir(exist_ok=True)
        
        self.checkpoint_file = self.model_output_dir / "checkpoint.jsonl"
        self.results_csv = self.model_output_dir / "detailed_results.csv"
        self.summary_txt = self.model_output_dir / "summary.txt"
        
        self.all_items: List[EvaluationItem] = []  # filled from checkpoint if resuming
        self.answer_extractor = AnswerExtractor()
        
        # load checkpoint
        self.processed_items = self._load_checkpoint()
    
    @staticmethod
    def _sanitize_model_name(name: str) -> str:
    # clear invalid characters
        return re.sub(r'[^\w\-]', '_', name)
    
    def _load_checkpoint(self) -> set:
        # restore processed indices and their results, so a resumed run reports complete statistics
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    self.all_items = [EvaluationItem(**json.loads(line)) for line in f if line.strip()]
                processed = {(it.language, it.index) for it in self.all_items}
                logger.info(f"Load checkpoint: {len(processed)} items have been processed")
                return processed
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        return set()
    
    def _save_checkpoint(self, item: EvaluationItem):
        # append-only: one finished item per line
        self.processed_items.add((item.language, item.index))
        try:
            with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_data(self) -> Dict[str, List[Dict]]:
        # load data for all languages

        data_by_language = {}
        languages = ["arabic", "chinese", "english", "hindi", "japanese", "spanish", "swahili"]
        
        for language in languages:
            # search for files
            files = list(self.data_dir.glob(f"{language}_*.json"))
            if not files:
                logger.warning(f"Failed to find files for {language}")
                continue
            
            # if multiple files, take the first one
            file_path = files[0]
            logger.info(f"Load {language} data from: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data_by_language[language] = data
                    logger.info(f"   - load {len(data)} data items")
            except Exception as e:
                logger.error(f"Failed to load data of {language}: {e}")
        
        return data_by_language
    
    def evaluate(self):
        logger.info("=" * 80)
        logger.info("starting evaluation")
        logger.info(f"model name: {self.model.model_name}")
        logger.info(f"output file: {self.model_output_dir}")
        logger.info("=" * 80)
        
        # load data first
        data_by_language = self.load_data()
        
        if not data_by_language:
            logger.error("No data loaded, exiting...")
            return
        
        # then process each language
        for language, data_list in data_by_language.items():
            logger.info(f"\nprocess {language} data...")
            self._process_language(language, data_list)
        
        # save
        self._save_results()
        
        self.model.close()
        
        logger.info("=" * 80)
        logger.info("Evaluation completed")
        logger.info("=" * 80)
    
    def _process_language(self, language: str, data_list: List[Dict]):
       # process data for a single language
        template = PROMPT_TEMPLATES[language]
        
        items_to_process = []
        for idx, item_data in enumerate(data_list):
            if (language, idx) not in self.processed_items:
                items_to_process.append((idx, item_data))

        # if all processed, skip
        if not items_to_process:
            logger.info(f"all items of {language} have been processed, skipping...")
            with tqdm(total=len(data_list), desc=f"process {language.ljust(8)}", unit="item") as pbar:
                pbar.update(len(data_list))
            return

        pbar = tqdm(
            items_to_process,
            total=len(items_to_process),
            desc=f"process {language.ljust(8)}", 
            unit="item"
        )

        for idx, item_data in pbar:
            
            try:
                # construct prompt
                prompt = template.format(
                    narrative=item_data['narrative'],
                    question=item_data['question']
                )
                
                response = self.model.generate(prompt)
                
                extracted = self.answer_extractor.extract_answer(response, language)
                
                is_valid = extracted is not None
                is_correct = False
                
                if is_valid:
                    # map to english
                    emotion_map = EMOTION_MAPPINGS[language]
                    expected_english = emotion_map.get(item_data['answer'], item_data['english_ver_answer'])
                    is_correct = (extracted == expected_english)
                
                # create evaluation item
                eval_item = EvaluationItem(
                    language=language,
                    index=idx,
                    narrative=item_data['narrative'],
                    question=item_data['question'],
                    answer=item_data['answer'],
                    english_ver_answer=item_data['english_ver_answer'],
                    model_response=response,
                    extracted_answer=extracted or "",
                    is_correct=is_correct,
                    is_valid=is_valid
                )
                
                self.all_items.append(eval_item)
                
                self._save_checkpoint(eval_item)
                
                time.sleep(REQUEST_INTERVAL)
                
            except Exception as e:
                logger.error(f"failed to process: {language}-{idx}: {e}")
                # load error item
                eval_item = EvaluationItem(
                    language=language,
                    index=idx,
                    narrative=item_data.get('narrative', ''),
                    question=item_data.get('question', ''),
                    answer=item_data.get('answer', ''),
                    english_ver_answer=item_data.get('english_ver_answer', ''),
                    model_response=f"ERROR: {str(e)}",
                    extracted_answer="",
                    is_correct=False,
                    is_valid=False
                )
                self.all_items.append(eval_item)
                self._save_checkpoint(eval_item)
    
    def _save_results(self):
        logger.info("\nSave results...")
        
        self._save_csv()
        
        self._save_summary()
        
        logger.info(f"Load results to: {self.model_output_dir}")
    
    def _save_csv(self):
        
        with open(self.results_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                'Language', 'Index', 'Narrative', 'Question', 'Answer', 
                'English Answer', 'Model Response', 'Extracted Answer',
                'Is Correct', 'Is Valid'
            ])
            
            for item in self.all_items:
                writer.writerow([
                    item.language,
                    item.index,
                    item.narrative,
                    item.question,
                    item.answer,
                    item.english_ver_answer,
                    item.model_response,
                    item.extracted_answer,
                    'Yes' if item.is_correct else 'No',
                    'Yes' if item.is_valid else 'No'
                ])
    
    def _save_summary(self):
        
        logger.info(f"Save summary: {self.summary_txt}")
        
        # summarize per language
        language_results = {}
        for language in PROMPT_TEMPLATES.keys():
            items = [item for item in self.all_items if item.language == language]
            
            if not items:
                continue
            
            total = len(items)
            correct = sum(1 for item in items if item.is_correct)
            invalid = sum(1 for item in items if not item.is_valid)
            valid_total = total - invalid
            
            accuracy = (correct / valid_total * 100) if valid_total > 0 else 0.0
            
            language_results[language] = LanguageResults(
                language=language,
                total=total,
                correct=correct,
                invalid=invalid,
                accuracy=accuracy
            )
        
        # overall results
        total_all = len(self.all_items)
        correct_all = sum(1 for item in self.all_items if item.is_correct)
        invalid_all = sum(1 for item in self.all_items if not item.is_valid)
        valid_total_all = total_all - invalid_all
        overall_accuracy = (correct_all / valid_total_all * 100) if valid_total_all > 0 else 0.0
        
        with open(self.summary_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Results for evaluation\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"model name: {self.model.model_name}\n")
            f.write(f"evaluation time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("params:\n")
            for key, value in GENERATION_CONFIG.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("Results for each language\n")
            f.write("-" * 80 + "\n\n")
            
            for language, result in sorted(language_results.items()):
                f.write(f"Language: {language.upper()}\n")
                f.write(f"  total: {result.total}\n")
                f.write(f"  correct: {result.correct}\n")
                f.write(f"  invalid: {result.invalid}\n")
                f.write(f"  valid: {result.total - result.invalid}\n")
                f.write(f"  accuracy: {result.accuracy:.2f}%\n")
                f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("Overall results\n")
            f.write("-" * 80 + "\n\n")
            
            f.write(f"total: {total_all}\n")
            f.write(f"correct: {correct_all}\n")
            f.write(f"invalid: {invalid_all}\n")
            f.write(f"valid: {valid_total_all}\n")
            f.write(f"accuracy: {overall_accuracy:.2f}%\n\n")
            
        
        logger.info("\n" + "=" * 80)
        logger.info("Overall Results:")
        logger.info("=" * 80)
        
        for language, result in sorted(language_results.items()):
            logger.info(f"{language.upper()}: {result.accuracy:.2f}% "
                        f"({result.correct}/{result.total - result.invalid})")
        
        logger.info(f"\nOverall accuracy: {overall_accuracy:.2f}% "
                    f"({correct_all}/{valid_total_all})")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # basic params
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['vllm', 'api'],
                        help='mode: vllm or api')
    parser.add_argument('--data_dir', type=str,
                        default=str(REPO_ROOT / 'data' / 'text-only' / 'multilingual'),
                        help='dir containing {language}_*.json files')
    parser.add_argument('--output_dir', type=str, default='results/text-only',
                        help='path to the output results')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='sampling temperature (0.0 = greedy decoding)')
    parser.add_argument('--max_tokens', type=int, default=128,
                        help='max new tokens per response')
    
    # vllm params
    parser.add_argument('--model_path', type=str,
                        help='[vllm] model path')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='[vllm] tensor parallel size')
    
    # api params
    parser.add_argument('--base_url', type=str,
                        help='[api] base url')
    parser.add_argument('--api_key', type=str,
                        help='[api] api key')
    parser.add_argument('--model_name', type=str,
                        help='model name for api or vllm')
    
    args = parser.parse_args()
    
    GENERATION_CONFIG["temperature"] = args.temperature
    GENERATION_CONFIG["max_tokens"] = args.max_tokens
    set_seed(args.seed)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(Path(args.output_dir) / 'evaluation.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    '''
    Example usage
    For vLLM (offline inference):
    python evaluation/run_benchmark_textonly.py --mode vllm --model_path /path/to/your/model

    For an OpenAI-compatible API:
    python evaluation/run_benchmark_textonly.py --mode api \
        --base_url https://api.openai.com/v1 --api_key sk-xxx --model_name your_model_name
    '''

    if args.mode == 'vllm':
        if not args.model_path:
            parser.error("--model_path is needed for vllm mode")
        if not args.model_name:
            args.model_name = Path(args.model_path).name
    elif args.mode == 'api':
        if not args.base_url or not args.api_key or not args.model_name:
            parser.error("--base_url, --api_key, and --model_name are needed for api mode")
    
    logger.info("Initializing model...")
    
    if args.mode == 'vllm':
        if LLM is None or SamplingParams is None:
            parser.error("failed to load vllm lib")
            
        model = VLLMModel(
            model_path=args.model_path,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size
        )
    else:  # else for api
        model = APIModel(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name
        )
    
    # create evaluator
    evaluator = Evaluator(
        data_dir=args.data_dir,
        model=model,
        output_dir=args.output_dir
    )
    
    try:
        evaluator.evaluate()
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted")
        logger.info("Saving current progress successfully")
    except Exception as e:
        logger.error(f"error: {e}", exc_info=True)
        raise
    finally:
        model.close()


if __name__ == '__main__':
    main()