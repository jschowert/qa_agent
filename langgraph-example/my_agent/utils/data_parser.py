from typing import List, Dict, Any
from lxml import etree
from .get_data import get_elasticsearch_data

class DataParser:
    # XPath 규칙 정의 - 클래스 변수로 선언
    XPATH_RULES = {
        'invention_title': "//Description/InventionTitle/text() | //description/invention-title/text() | //ApplicationBody/InventionTitle/text() | //PCTApplicationBody/PCTInventionTitle/text()",
        'technical_field': "//Description/TechnicalField//text() | //description/technical-field//text() | //ApplicationBody/Disclosure/TechnicalField//text() | //PCTApplicationBody/PCTTechnicalField//text() | //Description/TECHNICAL_FIELD//text()",
        'background_art': "//ApplicationBody/Disclosure/BackgroundTech//text() | //Description/BackgroundArt//text() | //description/background-art//text() | //ApplicationBody/Disclosure/BackgroundTech//text() | //PCTApplicationBody/PCTBackgroundArt//text() | //ApplicationBody/Disclosure/InventionPurpose/BackgroundArt//text() | //Description/BACKGROUND_ART//text()",
        'disclosure': "//description/disclosure//text() | //ApplicationBody/Disclosure/InventDetailContent//text() | //PCTApplicationBody/PCTDisclosure//text()",
        'summary_invention': "//Description/Disclosure/InventionSummary//text() | //Description/InventionSummary//text() | //description/summary-of-invention//text() | //Description/SUMMARY_OF_INVENTION//text()",
        'tech_problem': "//Description/Disclosure/TechnicalProblem//text() | //Description/Disclosure/InventionPurpose//text() | //description/summary-of-invention/tech-problem//text() | //ApplicationBody/Disclosure/InventionContent/SolutionProblem//text() | //ApplicationBody/Disclosure/InventionPurpose/AbstractProblem//text() | //Description/SUMMARY_OF_INVENTION/TECH_PROBLEM//text()",
        'tech_solution': "//Description/Disclosure/TechnicalSolution//text() | //Description/Disclosure/InventionComposition//text() | //description/summary-of-invention/tech-solution//text() | //ApplicationBody/Disclosure/InventionContent/MeansProblemSolution//text() | //Description/SUMMARY_OF_INVENTION/TECH_SOLUTION//text()",
        'advantageous_effects': "//Description/Disclosure/AdvantageousEffects//text() | //description/summary-of-invention/advantageous-effects//text() | //ApplicationBody/Disclosure/InventionContent/Effectiveness//text() | //Description/SUMMARY_OF_INVENTION/ADVANTAGEOUS_EFFECTS//text()",
        'description_embodiments': "//description/description-of-embodiments//text() | //ApplicationBody/Disclosure/InventDetailContent//text()",
        'description': "//description"
    }

    @staticmethod
    def find_text_by_xpath(root: etree._Element, xpath: str) -> List[str]:
        """XPath 표현식에 해당하는 모든 텍스트를 찾아 반환합니다."""
        results = []
        for path in xpath.split('|'):
            path = path.strip()
            elements = root.xpath(path)
            for element in elements:
                if isinstance(element, str):
                    results.append(element.strip())
                else:
                    text = ' '.join(element.xpath('.//text()'))
                    if text.strip():
                        results.append(text.strip())
        return results

    @staticmethod
    def parse_xml_sections(description_text: str) -> Dict[str, List[str]]:
        """XML에서 모든 섹션의 텍스트를 추출합니다."""
        try:
            root = etree.fromstring(description_text.encode('utf-8'))
            sections = {}
            for key, xpath in DataParser.XPATH_RULES.items():
                sections[key] = DataParser.find_text_by_xpath(root, xpath)
            return sections
        except Exception as e:
            print(f"XML 파싱 중 오류 발생: {str(e)}")
            return {}

    @staticmethod
    def parse_for_summary(doc_id: str) -> Dict[str, str]:
        """요약을 위한 데이터 파싱"""
        try:
            data = get_elasticsearch_data(doc_id)
            if not data:
                return {"error": "데이터를 찾을 수 없습니다."}

            source = data[0].get("_source", {})
            
            # description 또는 descriptionLarge에서 데이터 추출
            description_text = None
            if source.get("description"):
                description_text = source["description"][0]
            elif source.get("descriptionLarge"):
                description_text = source["descriptionLarge"][0]
            
            if not description_text:
                return {"error": "description 또는 descriptionLarge 데이터를 찾을 수 없습니다."}

            # XML 파싱
            sections = DataParser.parse_xml_sections(description_text)

            # 주요 섹션 텍스트 추출
            section_texts = {
                "발명의 명칭": " ".join(sections.get("invention_title", [])),
                "기술 분야": " ".join(sections.get("technical_field", [])),
                "배경 기술": " ".join(sections.get("background_art", [])),
                "발명의 공개": " ".join(sections.get("disclosure", [])),
                "발명의 요약": " ".join(sections.get("summary_invention", [])),
                "기술적 과제": " ".join(sections.get("tech_problem", [])),
                "해결 방안": " ".join(sections.get("tech_solution", [])),
                "효과": " ".join(sections.get("advantageous_effects", []))
            }

            # 비어있는 섹션 수 계산
            empty_sections = sum(1 for text in section_texts.values() if not text.strip())
            
            # 4개 이상의 섹션이 비어있으면
            if empty_sections >= 4:
                print(f"[경고] {empty_sections}개의 섹션이 비어있습니다. description을 포함하여 재구성합니다.")
                
                # 비어있지 않은 섹션만 선택
                valid_sections = {k: v for k, v in section_texts.items() if v.strip()}
                
                # description 전체 텍스트 추출
                full_description = " ".join(sections.get("description", []))
                if full_description.strip():
                    valid_sections["발명의 상세한 설명"] = full_description
                
                return valid_sections
            
            return section_texts
            
        except Exception as e:
            print(f"Error in parse_for_summary: {str(e)}")
            return {"error": str(e)}

    

    @staticmethod
    def parse_for_default(doc_id: str) -> Dict[str, Any]:
        """기본 응답을 위한 데이터 파싱"""
        data = get_elasticsearch_data(doc_id)
        if not data:
            return {"error": "데이터를 찾을 수 없습니다."}
        return data[0]["_source"]

# ... (기존 DataParser 클래스 코드는 그대로 유지) ...

# 테스트 코드
if __name__ == "__main__":
    import json
    from pprint import pprint
    
    def run_parser_tests():
        # 테스트할 문서 ID
        test_id = "kr20240172646b1"
        
        print("\n" + "="*50)
        print("데이터 파서 테스트 시작")
        print("="*50)

        # 1. Summary 파싱 테스트
        print("\n[요약 파싱 테스트]")
        summary_result = DataParser.parse_for_summary(test_id)
        print("\n요약 파싱 결과:")
        for key, value in summary_result.items():
            print(f"\n{key}:")
            print(f"{value[:200]}..." if len(value) > 200 else value)

        # 2. Purpose 파싱 테스트
        print("\n" + "="*50)
        print("\n[목적 파싱 테스트]")
        purpose_result = DataParser.parse_for_purpose(test_id)
        print("\n목적 파싱 결과:")
        for key, value in purpose_result.items():
            print(f"\n{key}:")
            print(f"{value[:200]}..." if len(value) > 200 else value)

        # 3. 상세 테스트 - XML 파싱 및 XPath 규칙 확인
        print("\n" + "="*50)
        print("\n[상세 테스트]")
        
        data = get_elasticsearch_data(test_id)
        if data and data[0].get("_source", {}).get("description"):
            description = data[0]["_source"]["description"][0]
            try:
                root = etree.fromstring(description.encode('utf-8'))
                print("\nXML 파싱: 성공")
                
                print("\nXPath 규칙 테스트 결과:")
                for key, xpath in DataParser.XPATH_RULES.items():
                    results = DataParser.find_text_by_xpath(root, xpath)
                    print(f"\n{key}:")
                    print(f"- 매칭된 요소 수: {len(results)}")
                    if results:
                        print(f"- 첫 번째 매칭 샘플: {results[0][:100]}...")
            
            except Exception as e:
                print(f"XML 파싱 실패: {str(e)}")
        
        # 4. 결과 저장
        results = {
            "summary": summary_result,
            "purpose": purpose_result
        }
        
        output_file = 'parser_test_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*50)
        print(f"테스트 결과가 {output_file} 파일에 저장되었습니다.")
        print("="*50)

    # 테스트 실행
    run_parser_tests()