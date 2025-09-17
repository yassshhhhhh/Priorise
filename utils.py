import json

data_dictt = {'Region': 'Geography for the data (Country/ Rural-urban/ State levels)',
 'Retainers%': 'HH buying brand in both time period T1 & T2\u200b',
 'Lapsers%': 'HH buying brand in time period T1 but not in T2\u200b',
 'New_Triers%': 'HH not buying brand in time period T1 but buying in T2\u200b',
 'Year': 'Moving Annual Total for the year of measurement ',
 'Level': 'Data granularity ',
 'Category': 'FMCG Category ',
 'Segment': 'Pack segment (like Bottles, Bars, etc.)',
 'Brand': 'Brand ',
 'Subbrand': 'Sub-brands within each brand ',
 'Item': 'Brand X Pack Size X Pack type ',
 'MedicinalForm': 'Medicinal Form in which the item falls under',
 'PackType': 'Packtypes within each Segment',
 'Purpose': 'Benefit of particular item for consumer (example - Fairness, Smoothness, etc.)',
 'Users': 'User type (e.g. Family/ Individuals, etc.)',
 'Fragrance_Ingredients': 'Key ingredient or fragrance used in the item ',
 'Price_Per_Sales_Unit': 'Price for every unit sold of an item',
 'Number_of_Stores_Retailing ': 'No. of stores retailing the item ',
 'Households': 'Absolute households purchasing the brand ',
 'Households_Growth%': 'Absolute households uplift or decline vs. last period ',
 'Penetration%': 'Reach of the brands/ pack - %of households purchasing the product',
 'Sales Value (In Cr)': 'Total value brought for the brand/packs',
 'Weighted Distribution Handling ': 'Weighted distribution for stock from manufacturing to reaching the consumers with respect to total category handling ',
 "Sales Volume (in '000 litres)": 'Total volume brought for the brand/packs'}

FAQs = {
    'CMI': [
        "How is the soap category performing?",
        "What is the performance of Aloe Vera fragrance soaps?",
        "How is Brand 1's performance?",
        "How is Brand 1 performing against other competition?"
    ],
    'Brand Manager': [
        "How is Brand 1's performance?",
        "How is Brand 1 performing against other competition?",
        "How is Brand 1 packtype sample bar within herbal category performing?",
        'How is HUL faring against competitors in Personal care segment?',
        'Please share all brand insights within Beverages along with a summary',
        'Which Personal care brands are not faring well and why?'
    ],
    'Category Manager': [
        "How is the soap category performing?",
        "What is the performance of Aloe Vera fragrance soaps"
        'How is rural India compared to Urban India in personal care?',
        'Which categories are big bets & where do we need to be cautious about?',
        'What are the growth drivers of Personal care category?',
        'What is the category distribution within Personal care consumption in last 3 years?'
    ]
}

mark_D = """
    <style>
        /* Hide file size limit text */
        .st-emotion-cache-7oyrc {
            display: none !important;
        }
        /* Additional backup selectors */
        .uploadedFile {
            display: none;
        }
        .stFileUploader > div > small {
            display: none !important;
        }
        small[class^='st-emotion-cache'] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        }
        .logo {
            width: 150px; 
            margin-bottom: lever10px;
        }
        </style>
    </style>
"""

dril_ppt = """
            Given the main query: "{query}", generate exactly three detailed drill-down questions that analyze brand metrics data. The data contains the following columns:
            - Brand
            - Date (Aug'23, Aug'24)
            - TOM (Top of Mind)
            - Spontaneous_Awareness
            - MOUB (Mention of Used Brand)
            - Purchase_6M (Purchase in last 6 months)
            - Purchase_1Y (Purchase in last 1 year)
            - Ever_Tried
            - Total_Awareness
            - Conv_Rate_MOUB%
            - Conv_rate_Purchase6M% (Conversion rate for 6M purchases)
            - Conv_rate_Purchase1Y% (Conversion rate for 1Y purchases)
            - Conv_rate_EverTried% (Conversion rate for Ever Tried)
            Generate questions that:
            1. Compare year-over-year changes in key metrics
            2. Analyze brand performance across different awareness and purchase metrics
            3. Identify brand strengths and weaknesses in the awareness funnel and conversion rates
            Format the output as a Python list of strings, similar to the following example and nothing else apart from this list:
            [  
            "What is the year-over-year change in TOM and Spontaneous Awareness for each brand?",  
            "Compare the conversion rates (Conv_Purchase6M, Conv_Purchase1Y) across years (2023 & 2024)",  
            "Analyze the complete awareness funnel (Total Awareness to TOM)"  
            ] 
            
            Now, generate similar drill-down questions for the main query: "{query}".
            """

drill_p_m = """
                Given the main query: "{query}", generate exactly six detailed drill-down questions that analyze the performance, penetration, and key contributors based on the following data schema:
                Data Schema:
                {data_dictt}
                Note: All text values in the dataset are lowercase (e.g., 'brand 1', 'sample bar', 'herbal').
                Unique Values for Object Columns:
                {unique_values_dict}
                Instructions:
                1. Use the schema to identify column names (e.g., 'Brand', 'PackType') and their meanings.
                2. Extract specific column-value pairs from the query (e.g., 'Brand' = 'Brand 1', 'PackType' = 'sample bar', 'MedicinalForm' = 'herbal'), lowercase all extracted values (e.g., 'brand 1', 'sample bar', 'herbal'), and ensure they match the unique values provided. If a value isn't in the unique values list, adjust to the closest valid match or skip it.
                3. Cover these categories:
                - Sales and revenue performance for 2024, 2023, 2022
                - Region-wise performance (strictly all india urban vs. all india rural) for 2024 only
                - Penetration for 2024 only
                - Segment behavior for 2024 only
                - Competitive analysis for 2024 only
                - Weighted Distribution Handling for 2024 only
                4. Use exact column names from the schema with backticks if they contain spaces (e.g., `Sales Value (In Cr)`).
                5. Avoid quarterly analysis.
                6. Format the output as a Python list of strings, matching the structure of this example:
                Example for query "How is Brand 1 packtype sample bar within herbal category performing?":
                [
                "What were the total `Sales Value (In Cr)` and `Sales Volume (in '000 litres)` for `Brand` = 'brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal' across 2024, 2023, and 2022?",
                "Calculate the `Sales Value (In Cr)` for each year and determine the percentage growth or decline between 2024 vs. 2023 and 2023 vs. 2022 for `Brand` = 'brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal'.",
                "In 2024, what is the `Sales Value (In Cr)` for `Brand` = 'brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal' in all india rban vs. all india rural?",
                "For 2024, what is the `Penetration%` for `Brand` = 'brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal'?",
                "In 2024, what are the `Retainers%`, `Lapsers%`, and `New_Triers%` for `Brand` = 'brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal'?",
                "In 2024, how does the `Weighted Distribution Handling` support `Brand` = 'brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal' and it's leading `Segment`?"
                ]
                Now, generate similar drill-down questions for the main query: "{query}".
                """


consolidation_prompt = """
                        Based on the original query: "{temp_query}" and the following drill-down analysis results:
                        {dril_results_PPT},
                        please provide a consolidated, holistic insight (strictly along with numbers, and no insights that is general) that integrates all these data cuts. And if you see error message in the drill results do not include that.
                        Hygiene Factors:
                        1. Decimal points up to a maximum of 2 for precision.
                        2. All metrics should include units (e.g., INR, units, %, etc.).
                        3. Sales value should be in Cr (INR Crores). The sales value is already in Cr, no need to convert explicitly
                        4. Mention months and timeframes wherever data allows for trend analysis.
                        5. Regional Performance (2024) - only All India Rural Vs All India Urban comparision
                        6. If the data is not explicitly provided do not call it out in the insights instead provide what is present along with numbers
                        Example consolidation format:
                        Final Consolidated Insight
                            Based on the provided data, here is a consolidated and holistic insight into the performance of Brand 1:
                            1. Overall Performance (2022-2024)
                                Sales Value: Brand 1's sales value increased from INR 95.74 Cr in 2022 to INR 103.42 Cr in 2024, marking a growth of 8.03% over the two years.
                                Sales Volume: The sales volume grew from 59,070.44 million litres in 2022 to 61,549.85 million litres in 2024, indicating a 4.19% increase.
                                Year-over-Year (YoY) Changes:
                                2023: Sales value increased by 4.52%, the sales value is 2922 Cr and sales volume by 0.67%.
                                2024: Sales value increased by 3.36%, the sales value is 2933 Cr and sales volume by 3.51%.
                            2. Regional Performance (2024)
                                Rural: Contributed INR 23.99 Cr.
                                Urban: Contributed INR 15.43 Cr.
                            3. Market Penetration (2024)
                                Penetration Rate: 0.06%
                            4. Consumer Analysis (2024):
                                Retainers: 69%
                                Lapsers: 31%
                                New Triers: 32%
                            5. Distribution (2024)
                                Brand 1: The weighted distribution for Brand 1 was 1781.17.
                                Segment: Sample Bar, had the highest revenue contribution of 8098.20 Cr.
                        
                        Example for "How is Brand 1 against competition?":
                        Final Consolidated Insight
                            Based on the provided data, here is a consolidated insight for "How is Brand 1 against competition?":
                            1. Sales Performance (2022-2024)
                                Brand 1: Sales value increased from INR 95.74 Cr in 2022 to INR 103.42 Cr in 2024, a growth of 8.03%; volume from 59,070.44 '000 litres to 61,549.85 '000 litres, up 4.19%.
                                Brand 2: Sales value increased from INR 80.50 Cr in 2022 to INR 85.20 Cr in 2024, a growth of 5.84%; volume from 50,000.00 '000 litres to 52,100.00 '000 litres, up 4.20%.
                            2. Year-over-Year Changes (2022-2024)
                                Brand 1: 2023 vs. 2022: Sales value up 4.52% to INR 100.07 Cr, volume up 0.67%; 2024 vs. 2023: up 3.36% to INR 103.42 Cr, volume up 3.51%.
                                Brand 2: 2023 vs. 2022: Sales value up 3.11% to INR 83.00 Cr, volume up 2.00%; 2024 vs. 2023: up 2.65% to INR 85.20 Cr, volume up 2.16%.
                            3. Regional Performance (2024)
                                Brand 1: Rural: INR 23.99 Cr, Urban: INR 15.43 Cr
                                Brand 2: Rural: INR 20.50 Cr, Urban: INR 13.70 Cr
                            4. Market Penetration (2024)
                                Brand 1: 0.06%
                                Brand 2: 0.04%
                            5. Competitive Analysis (2024)
                                Brand 1 vs. Brand 2: 'Brand 1' achieved INR 103.42 Cr in sales and 0.06% penetration, outperforming 'Brand 2' with INR 85.20 Cr and 0.04%.
                            6. Distribution (2024)
                                Brand 1: Weighted Distribution 1781.17
                                Brand 2: Weighted Distribution 1450.306. Distribution (2024)
                        """
consolidatin_prompt_main = """
                        Based on the original query: "{temp_query}" and the following drill-down analysis results:
                        {drill_results},
                        please provide a consolidated, holistic insight (strictly along with numbers, and no insights that is general) that integrates all these data cuts. And if you see error message in the drill results do not include that.
                        Hygiene Factors:
                        1. Decimal points up to a maximum of 2 for precision.
                        2. All metrics should include units (e.g., INR, units, %, etc.).
                        3. Sales value should be in Cr (INR Crores). The sales value is already in Cr, no need to convert explicitly
                        4. Mention months and timeframes wherever data allows for trend analysis.
                        5. Regional Performance (2024) - only All India Rural Vs All India Urban comparision
                        6. If the data is not explicitly provided do not call it out in the insights instead provide what is present along with numbers
                        Example consolidation format:
                        Final Consolidated Insight
                            Based on the provided data, here is a consolidated and holistic insight into the performance of Brand 1:
                            1. Overall Performance (2022-2024)
                                Sales Value: Brand 1's sales value increased from INR 95.74 Cr in 2022 to INR 103.42 Cr in 2024, marking a growth of 8.03% over the two years.
                                Sales Volume: The sales volume grew from 59,070.44 million litres in 2022 to 61,549.85 million litres in 2024, indicating a 4.19% increase.
                                Year-over-Year (YoY) Changes:
                                2023: Sales value increased by 4.52%, the sales value is 2922 Cr and sales volume by 0.67%.
                                2024: Sales value increased by 3.36%, the sales value is 2933 Cr and sales volume by 3.51%.
                            2. Regional Performance (2024)
                                Rural: Contributed INR 23.99 Cr.
                                Urban: Contributed INR 15.43 Cr.
                            3. Market Penetration (2024)
                                Penetration Rate: 0.06%
                            4. Consumer Analysis (2024):
                                Retainers: 69%
                                Lapsers: 31%
                                New Triers: 32%
                            5. Distribution (2024)
                                Brand 1: The weighted distribution for Brand 1 was 1781.17.
                                Segment: Sample Bar, had the highest revenue contribution of 8098.20 Cr.
                        
                        Example for "How is Brand 1 against competition?":
                        Final Consolidated Insight
                            Based on the provided data, here is a consolidated insight for "How is Brand 1 against competition?":
                            1. Sales Performance (2022-2024)
                                Brand 1: Sales value increased from INR 95.74 Cr in 2022 to INR 103.42 Cr in 2024, a growth of 8.03%; volume from 59,070.44 '000 litres to 61,549.85 '000 litres, up 4.19%.
                                Brand 2: Sales value increased from INR 80.50 Cr in 2022 to INR 85.20 Cr in 2024, a growth of 5.84%; volume from 50,000.00 '000 litres to 52,100.00 '000 litres, up 4.20%.
                            2. Year-over-Year Changes (2022-2024)
                                Brand 1: 2023 vs. 2022: Sales value up 4.52% to INR 100.07 Cr, volume up 0.67%; 2024 vs. 2023: up 3.36% to INR 103.42 Cr, volume up 3.51%.
                                Brand 2: 2023 vs. 2022: Sales value up 3.11% to INR 83.00 Cr, volume up 2.00%; 2024 vs. 2023: up 2.65% to INR 85.20 Cr, volume up 2.16%.
                            3. Regional Performance (2024)
                                Brand 1: Rural: INR 23.99 Cr, Urban: INR 15.43 Cr
                                Brand 2: Rural: INR 20.50 Cr, Urban: INR 13.70 Cr
                            4. Market Penetration (2024)
                                Brand 1: 0.06%
                                Brand 2: 0.04%
                            5. Competitive Analysis (2024)
                                Brand 1 vs. Brand 2: 'Brand 1' achieved INR 103.42 Cr in sales and 0.06% penetration, outperforming 'Brand 2' with INR 85.20 Cr and 0.04%.
                            6. Distribution (2024)
                                Brand 1: Weighted Distribution 1781.17
                                Brand 2: Weighted Distribution 1450.306. Distribution (2024)
                        """


            # drill_prompt_main = f"""
            # Given the main query: "{query}", generate exactly six detailed drill-down questions that analyze the performance, penetration, and key contributors. Ensure the questions cover
            # 1. Sales and revenue performance in year 2024, 2023, 2022
            # 2. Region-Wise, All India Urban Vs All India Rural performance for 2024 only 
            # 3. Penetration for 2024 only
            # 4. Segment behavior for 2024 only
            # 5. Competitive analysis for 2024 only
            # 6. Weighted Distribution Handling for 2024 only
            # 7. Avoid quarterly analysis 
            # 8. Strictly follows the Data Schema - {data_dict}: to differentiate the column names and values from the {query}
            
            # Format the output as a Python list of strings, similar (do not generalise) to the following example and nothing else apart from this list:
            
            #  [
            # "What were the total Sales Value (In Cr) and Sales Volume (in '000 litres) for Brand 1 across 2024, 2023, and 2022?",
            # "Calculate the Sales Value (In Cr) for each year and determine the percentage growth or decline in sales and volume between 2024 vs. 2023 and 2023 vs. 2022.",
            # "In 2024, All India Urban Vs All India Rural Sales Value for Brand 1?",
            # "For 2024, what is the Penetration% of Brand 1?",
            # "In 2024, what are the Retainers%, Lapsers%, and New_Triers% for Brand 1?", 
            # "In 2024, how does the Weighted Distribution Handling support the Brand 1 and it's leading Segment",
            # ]
            
            # Now, generate similar drill-down questions for the main query: "{query}".
            # """
            #powerful multi-column and non multi-column drill
            # drill_prompt_main = f"""
            # Given the main query: "{query}", generate exactly six detailed drill-down questions that analyze the performance, penetration, and key contributors based on the following data schema:

            # Data Schema:
            # {json.dumps(data_dict, indent=2)}
            # Note: All text values in the dataset are lowercase (e.g., 'brand 1', 'sample bar', 'herbal', 'soap', 'aloe vera').
            # Instructions:
            # 1. Use the schema to identify column names (e.g., 'Brand', 'PackType') and their meanings.
            # 2. Extract specific column-value pairs from the query (e.g., 'Brand' = 'Brand 1', 'PackType' = 'sample bar', 'MedicinalForm' = 'herbal') and apply these filters consistently across all questions.
            # 3. Cover these categories:
            # - Sales and revenue performance for 2024, 2023, 2022
            # - Region-wise performance (All India Urban vs. All India Rural) for 2024 only
            # - Penetration for 2024 only
            # - Segment behavior for 2024 only
            # - Competitive analysis for 2024 only
            # - Weighted Distribution Handling for 2024 only
            # 4. Use exact column names from the schema with backticks if they contain spaces (e.g., `Sales Value (In Cr)`).
            # 5. Avoid quarterly analysis.
            # 6. Format the output as a Python list of strings, matching the structure of this example:

            # Example for query "How is Brand 1 packtype sample bar within herbal category performing?":
            # [
            # "What were the total `Sales Value (In Cr)` and `Sales Volume (in '000 litres)` for `Brand` = 'Brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal' across 2024, 2023, and 2022?",
            # "Calculate the `Sales Value (In Cr)` for each year and determine the percentage growth or decline between 2024 vs. 2023 and 2023 vs. 2022 for `Brand` = 'Brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal'.",
            # "In 2024, what is the `Sales Value (In Cr)` for `Brand` = 'Brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal' in All India Urban vs. All India Rural?",
            # "For 2024, what is the `Penetration%` for `Brand` = 'Brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal'?",
            # "In 2024, what are the `Retainers%`, `Lapsers%`, and `New_Triers%` for `Brand` = 'Brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal'?",
            # "In 2024, how does the `Weighted Distribution Handling` support `Brand` = 'Brand 1', `PackType` = 'sample bar', and `MedicinalForm` = 'herbal' compared to its leading `Segment`?"
            # ]

            # Now, generate similar drill-down questions for the main query: "{query}".
            # """