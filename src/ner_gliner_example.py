import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

#urchade/gliner_multi-v2.1
#urchade/gliner_medium-v2.1
#urchade/gliner_multi_pii-v1

#text = "Maria Rodriguez loves making sushi and pizza in her spare time. She practices yoga and rock climbing on weekends in Boulder, Colorado."

#text = "47 anos. paciente extabagista, obesa grau iii, portadora de hipotireoidismo e insuficiência cardíaca com fração de ejeção reduzida 30%, "
#doença arterial coronária?. abertura: 06/06/2020, alta: 05/07/2020, estadia: 29 d 19 h, leito: enfermaria, convênio: sus. admitida neste serviço no dia 06/06 com quadro compatível com coronavírus  19. evolução com necessidade de múltiplas manobras de prona devido hipoxemia refratária. evolução com múltiplas infecções nosocomiais, choque séptico refratário, lesão renal aguda com necessidade de hemodiálise e disfunção múltipla de órgãos. sintomas iniciais com tosse seca e febre e mal estar geral evolução com síndrome respiratória aguda grave com necessidade de intubação orotraqueal e ventilação mecânica. hemocultura positiva para acinetobacter baumannii multirresistente. cateter triplo lúmen em veia subclávia esquerda pressão arterial invasiva em artéria radial esquerda cateter central de inserção periférica em membro superior esquerdo cateter de hemodiálise em veia jugular interna direita traqueostomia. parada cardiorrespiratória em assistolia sendo constatado óbito às {omitido} do dia {omitido}
#"
text = "Cristiano Ronaldo dos Santos Aveiro, born 5 February 1985, is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time."

labels = ["person", "award", "date", "competitions", "teams"]

#text = "[person], 47 years old. ex-smoker, obese grade iii, with hypothyroidism and heart failure with reduced ejection fraction 30%, coronary artery disease?. admission: 06/06/2020, discharge: 05/07/2020, stay: 29 d 19 h, location: ward, payer: sus. admitted to this service on 06/06 with a condition compatible with coronavirus 19. evolution with the need for multiple proning maneuvers due to refractory hypoxemia. evolution with multiple nosocomial infections, refractory septic shock, acute kidney injury requiring hemodialysis, and multiple organ dysfunction. initial symptoms with dry cough and fever and general malaise evolution with severe acute respiratory syndrome requiring orotracheal intubation and mechanical ventilation. positive blood culture for multidrug-resistant acinetobacter baumannii. triple lumen catheter in the left subclavian vein invasive blood pressure in the left radial artery peripherally inserted central catheter in the left upper limb hemodialysis catheter in the right internal jugular vein tracheostomy. cardiorespiratory arrest in asystole with death confirmed at on the day"

# Try with simpler, English labels first
#labels = ["Person", "Food", "Hobby", "Location"]
#labels = ["Hospitalizacao", "Data_Entrada", "Data_Saida", "Dias_Hospitalizado", "obito", "idade", "sexo", "Sintomas"]   
#labels = ['Hospitalization', 'Admission_Date', 'Discharge_Date', 'Days_Hospitalized', 'Death', 'Age', 'Sex', 'Symptoms']

# Using inference with proper batch handling
#entities = model.inference([text], labels)
entities = model.predict_entities(text, labels, threshold=0.5)

#entities = model.predict_entities(text, labels)

#for entity in entities:
#   print(entity["text"], "=>", entity["label"])

#print("Result type:", type(entities))
#print("Result:", entities)

if entities and len(entities) > 0:
    for entity in entities:
        print(entity['text'], "=>", entity['label'])
else:
    print("No entities found")