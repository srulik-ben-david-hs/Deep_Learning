from transformers import pipeline
from gimme.hs_gimme.constants import professions
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

sequence_to_classify = """ Developing, editing content as per guidelines prescribed by concerned governing educational authorities for CBSE 10th Standard English and Economics.
 Collection, editing and proof reading of content as per university and board guidelines.
1. Crisis Management
2. The Impact of Cross cultural factors on the
life-cycle of Product.
3. Green Marketing
•
Guest faculty at SIES across Mumbai for
undergraduate and post graduate courses.
•
Areas: Insurance and Introduction to Marketing,
Soft Skills.
"""
candidate_labels_org = ['Accounting_&_Finance', 'Administrative', 'Banking', 'Building_&_Construction', 'Business_&_Strategic_Management', 'Customer_Support', 'Design_&_Creative', 'Editorial_&_Content', 'Education', 'Biology', 'Engineering', 'Food_&_Hospitality', 'Human_Resources', 'Insurance', 'IT_&_Software_Development', 'Legal', 'Logistics_&_Transportation', 'Maintenance_&_Repair', 'Manufacturing', 'Marketing', 'Medical', 'Quality_Assurance', 'Real_Estate', 'Sales', 'Security', 'Supply_Chain_Management']
candidate_labels = ['Accounting and Finance', 'Administrative', 'Banking', 'Building and Construction', 'Business and Strategic Management', 'Customer Support', 'Design and Creative', 'Editorial and Content', 'Education', 'Biology', 'Engineering', 'Food and Hospitality', 'Human and Resources', 'Insurance', 'IT and Software Development', 'Legal', 'Logistics and Transportation', 'Maintenance and Repair', 'Manufacturing', 'Marketing', 'Medical', 'Quality Assurance', 'Real Estate', 'Sales', 'Security', 'Supply Chain Management']
cls = classifier(sequence_to_classify, candidate_labels)
breakpoint()

def profession_labels():
    prof = professions.PROFESSIONS_MAPPING
    prof_values=list(prof.values())
    main_prof = list(set([i.split('#')[0] for i in prof_values]))
    main_prof = [i.replace('_&_', ' ').replace('_', ' ') for i in main_prof]