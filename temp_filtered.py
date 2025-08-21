import openai
import os
import re
from typing import List
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
import psycopg2 

load_dotenv()
   
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Postgres Setup ===
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)
cursor = conn.cursor()

INTENT_PROMPT = """

Classify the user's prompt into one of the following intents:
- simple: a direct lookup or factual question about a single entity or KPI, without time-based or multi-metric comparison
- analysis: questions that involve time ranges, trend analysis, or metrics calculation
- compare: questions that compare two or more entities, SKUs, categories, or time periods

Respond with only one word: simple, analysis, or compare.


Prompt: "{user_prompt}"
"""

# For Testing purpose

queries_data = {
    "queries": [
        {
        "question": "What is the Household Penetration Growth for Brand 1 with a pack size of 180 over the past year?",
        "kpi": "Household Penetration Growth",
        "filters": {
            "brand_filtered": "Brand 1",
            "pack_size": 180,
            "period_order": "ascending"
        },
        "sql": '''SELECT
        "brand_filtered",
        "pack_size",
        "period",
        ("penetration_percentage" - LAG("penetration_percentage", 1) OVER (PARTITION BY "brand_filtered", "pack_size" ORDER BY "period")) * 100 AS "Household Penetration Growth"
        FROM
        "PRI_iop_hh_consumption"
        WHERE
        "brand_filtered" = 'Brand 1' AND
        "pack_size" = 180
        ORDER BY
        "period" ASC; '''
        },
        {
        "question": "What is the Share Change Value for brands with aloevera fragrance over the last year?",
        "kpi": "Share Change Value",
        "filters": {
            "fragrance_ingredients": "ALOE VERA",
            "category": "SOAP",
            "periods": ["MAT OCT'23", "MAT OCT'22"]
        },
        "sql": '''
        SELECT
        brand,
        ((Brand_Value_This_Year / NULLIF(Category_Value_This_Year, 0) * 100) - (Brand_Value_Last_Year / NULLIF(Category_Value_Last_Year, 0) * 100)) AS Share_Change_Value
        FROM
        (
        SELECT
        brand,
        SUM(CASE WHEN period = 'MAT OCT''23' THEN sales_value ELSE 0 END) AS Brand_Value_This_Year,
        SUM(CASE WHEN period = 'MAT OCT''22' THEN sales_value ELSE 0 END) AS Brand_Value_Last_Year,
        SUM(CASE WHEN period = 'MAT OCT''23' AND category = 'SOAP' THEN sales_value ELSE 0 END) AS Category_Value_This_Year,
        SUM(CASE WHEN period = 'MAT OCT''22' And category = 'SOAP' THEN sales_value ELSE 0 END) AS Category_Value_Last_Year
        FROM
        raw_sales_rms
        WHERE
        fragrance_ingredients = 'ALOE VERA'
        GROUP BY
        brand
        ) AS subquery
        '''
        },
        {
        "question": "How has the Share Change Value of Brand 1 with a pack size of 180 changed in the last year?",
        "kpi": "Share Change Value",
        "filters": {
            "brand": "Brand 1",
            "basepacksize": 180,
            "category": "SOAP",
            "periods": ["MAT OCT'23", "MAT OCT'22"]
        },
        "sql": '''
        SELECT
        ((Brand1_Value_This_Year / NULLIF(Category_Value_This_Year, 0) * 100) - (Brand1_Value_Last_Year / NULLIF(Category_Value_Last_Year, 0) * 100)) AS Share_Change_Value
        FROM
        (
        SELECT
        SUM(sales_value) AS Brand1_Value_This_Year,
        (SELECT SUM(sales_value) FROM "raw_sales_rms" WHERE period = 'MAT OCT''23' AND category = 'SOAP') AS Category_Value_This_Year
        FROM "raw_sales_rms"
        WHERE brand = 'Brand 1' AND basepacksize = 180 AND period = 'MAT OCT''23'
        ) AS This_Year,
        (
        SELECT
        SUM(sales_value) AS Brand1_Value_Last_Year,
        (SELECT SUM(sales_value) FROM "raw_sales_rms" WHERE period = 'MAT OCT''22' AND category = 'SOAP') AS Category_Value_Last_Year
        FROM "raw_sales_rms"
        WHERE brand = 'Brand 1' AND basepacksize = 180 AND period = 'MAT OCT''22'
        ) AS Last_Year
        '''
        }
    ]    
}

kpi_dict = {
    "Sales Growth Percentage": {
        "definition": "Total sales growth percentage",
        "logic": "((SUM(CASE WHEN {period_col} = '{end_period}' THEN sales_value ELSE 0 END) - " 
                 "SUM(CASE WHEN {period_col} = '{start_period}' THEN sales_value ELSE 0 END)) / " 
                 "NULLIF(SUM(CASE WHEN {period_col} = '{start_period}' THEN sales_value ELSE 0 END),0)) * 100",
        "tables": ["raw_sales_rms"],
        "keywords": ["sales", "growth", "percentage", "performance"]
    },
    "Volume Growth Percentage": {
        "definition": "Total volume growth percentage",
        "logic": "((SUM(CASE WHEN {period_col} = '{end_period}' THEN sales_vol ELSE 0 END) - " 
                 "SUM(CASE WHEN {period_col} = '{start_period}' THEN sales_vol ELSE 0 END)) / " 
                 "NULLIF(SUM(CASE WHEN {period_col} = '{start_period}' THEN sales_vol ELSE 0 END),0)) * 100",
        "tables": ["raw_sales_rms"],
        "keywords": ["volume", "growth", "percentage", "performance"]
    },
    "Unit Growth Percentage": {
        "definition": "Total Unit growth percentage",
        "logic": "((SUM(CASE WHEN {period_col} = '{end_period}' THEN sales_units ELSE 0 END) - " 
                 "SUM(CASE WHEN {period_col} = '{start_period}' THEN sales_units ELSE 0 END)) / " 
                 "NULLIF(SUM(CASE WHEN {period_col} = '{start_period}' THEN sales_units ELSE 0 END),0)) * 100",
        "tables": ["raw_sales_rms"],
        "keywords": ["unit", "growth", "percentage", "performance"]
    },
    "Share Change Value": {
        "definition": "Share Change Value",
        "logic": (
            """(
            (SUM(sales_value) FILTER (WHERE {period_col} = '{end_period}' {brand_filter} {fragrance_filter} {market_filter}) /
            NULLIF(SUM(sales_value) FILTER (WHERE {period_col} = '{end_period}' {category_filter}), 0) * 100)
            -
            (SUM(sales_value) FILTER (WHERE {period_col} = '{start_period}' {brand_filter} {fragrance_filter} {market_filter}) /
            NULLIF(SUM(sales_value) FILTER (WHERE {period_col} = '{start_period}' {category_filter}), 0) * 100)
            )""",
        ),
        "tables": ["raw_sales_rms"],
        "keywords": ["share", "change", "value", "category", "market"]
    },
    "Share Change Volume": {
        "definition": "Share Change Volume",
        "logic": (
            """(
            (SUM(sales_vol) FILTER (WHERE {period_col} = '{end_period}' {brand_filter} {fragrance_filter} {market_filter}) /
            NULLIF(SUM(sales_vol) FILTER (WHERE {period_col} = '{end_period}' {category_filter}), 0) * 100)
            -
            (SUM(sales_vol) FILTER (WHERE {period_col} = '{start_period}' {brand_filter} {fragrance_filter} {market_filter}) /
            NULLIF(SUM(sales_vol) FILTER (WHERE {period_col} = '{start_period}' {category_filter}), 0) * 100)
            )""",
        ),
        "tables": ["raw_sales_rms"],
        "keywords": ["share", "change", "volume", "category", "market"]
    },
    "Household Penetration Growth": {
        "definition": "Difference in penetration percentage between two periods",
        "logic": ("""(
                (AVG(CASE WHEN period = '{end_period}' THEN penetration_percentage END))
                -
                (AVG(CASE WHEN period = '{start_period}' THEN penetration_percentage END))
            )"""),
        "tables": ["PRI_iop_hh_consumption"],
        "keywords": ["household", "penetration", "growth", "points", "yearly change"]
    },
    # Sample: The Error I got Earlier where it's 3 aurguments 
    "Top Pack Size by Sales Volume": {
        "definition": "Lists the top-selling pack sizes by sales volume, optionally filtered by product/brand/item.",
        "logic": (
            "SELECT basepacksize, SUM(sales_vol) AS total_volume "
            "FROM raw_sales_rms "
            "WHERE {filters} "
        ),
        "tables": ["raw_sales_rms"],
        "keywords": ["top", "pack size", "selling", "most", "aloe vera", "soap", "volume"]
    },

}

# ----------------

schema_json = {

    "raw_sales_rms": {
        "columns": ["day_of_year", "sales_value", "sales_vol", "sales_units", "price_per_sales_unit", "wghtd_dist_handling", "relative_numeric_distribution_handling_product", 
                    "value_shr_in_handlers_product", "number_of_stores_retailing", "numeric_out_of_stock", "year", "month", "week", "quarter", "day", 
                    "day_of_week", "basepacksize", "level", "category", "segment", "manufacturer", "brand", "sub_brand", "item", "medicinal_form", "day_name", "month_name", "pack_type", 
                    "purpose", "users", "fragrance_ingredients", "period", "market"],
        "categorical_info": {
            "market": {
                "definition": "Geographic or retail area (e.g., All India, Urban + Rural)",
                "sample_values": ["All India Urban", "All India Rural", "Gujarat", "Karnataka", "Maharashtra", "Rajasthan", "UP"]
            },
            "level": {
                "definition": "Data level (e.g., item-level, brand-level)",
                "sample_values": ["ITEM"]
            },
            "category": {
                "definition": "Main product category (e.g., SOAP)",
                "sample_values": ["SOAP"]
            },
            "segment": {
                "definition": "Subdivision of the category (e.g., BOTTLES)",
                "sample_values": ["BOTTLES", "BARS"]
            },
            "manufacturer": {
                "definition": "Name of the producing company",
                "sample_values": ["ZXC", "wertjkl", "vjkop"]
            },
            "brand": {
                "definition": "Main brand name",
                "sample_values": ["Brand 1", "Brand 2", "Brand 3", "Brand 4", "Brand 5", "Brand 6", "Brand 7", "Brand 8"]
            },
            "sub_brand": {
                "definition": "Subdivision of brand",
                "sample_values": ["Brand 1 SubBrand 1", "Brand 1 SubBrand 2", "Brand 2 SubBrand 1", "Brand 3 SubBrand 1", "Brand 4 SubBrand 1", "Brand 5 SubBrand 1", "Brand 6 SubBrand 1", "Brand 7 SubBrand 1", "Brand 8 SubBrand 1"]
            },
            "item": {
                "definition": "Specific product SKU (includes weight/pack)",
                "sample_values": ["Brand 1 1000 GM BOTTLE", "Brand 2 50 GM CDBOX", "Brand 1 5 GM SAMPLE BAR"]
            },
            "medicinal_form": {
                "definition": "Nature of product (e.g., BEAUTY, MEDICINAL)",
                "sample_values": ["BEAUTY", "AYURVEDIC", "UNSPECIFIED", "HERBAL"]
            },
            "basepacksize": {
                "definition": "Size of pack (e.g., 180 gm)",
                "sample_values": ["1000", "180", "340", "440", "640", "80", "4.5", "3.5", "3.6", "4.6"]
            },
            "pack_type": {
                "definition": "Type of pack (e.g., BOTTLE, BOX)",
                "sample_values": ["BOTTLE", "CDBOX", "TUBES", "SAMPLE BAR"]
            },
            "purpose": {
                "definition": "Intended benefit (e.g., FAIRNESS, MOISTURIZING)",
                "sample_values": ["FAIRNESS", "MOISTURIZING", "DRY SKIN", "SOFT SKIN"]
            },
            "users": {
                "definition": "Target audience (e.g., FAMILY, MEN)",
                "sample_values": ["FAMILY", "UNSPECIFIED", "ADULT/FAMILY"]
            },
            "fragrance_ingredients": {
                "definition": "Fragrance type or if unspecified",
                "sample_values": ["ALOE VERA", "MUSK", "JASMINE", "UNSPECIFIED", "GARDENIA", "VETIVER", "LAVENDER", "ORANGE BLOSSOM", "BASIL", "PEONY", "HONEYSUCKLE", "ROSE", "APPLE", "TUBEROSE", "LILAC", "CHAMOMILE", "OCEAN BREEZE"]
            },
            "period": {
                "definition": "Reporting time period (e.g., MAT OCT'23)",
                "sample_values": ["MAT OCT'22", "MAT OCT'23", "MAT OCT'24"]
            }, 
            "sales_value": {
                "definition": "Revenue in currency (₹ or other), calculated as sales_unit*price_unit",
                "sample_values": [11.513, 290.065, 3.829]
            }, 
            "sales_vol": {
                "definition": "Volume sold — likely in liters (or ml, but data suggests liters)",
                "sample_values": [20.723, 122.14, 95.746]
            }, 
            "sales_units": {
                "definition": "Number of items sold",
                "sample_values": [0.115, 0.532, 5.543]
            }, 
            "price_per_sales_unit": {
                "definition": "Revenue per item (SALES_VALUE / SALES_UNITS)",
                "sample_values": [100, 105.421, 170.01]
            }, 
            "wghtd_dist_handling": {
                "definition": "Percentage of stores that a product is sold in, weighted by Category Sales",
                "sample_values": [0.004, 0.198, 2.24]
            }, 
            "relative_numeric_distribution_handling_product": {
                "definition": "Percentage of stores that sell a product",
                "sample_values": [0.001, 0.008, 0.23]
            }, 
            "value_shr_in_handlers_product": {
                "definition": "Share of product value in stores where it's available",
                "sample_values": [0.6802721, 0.0446429, 3.2967033]
            }, 
            "number_of_stores_retailing": {
                "definition": "Number of stores selling this product",
                "sample_values": [47.333, 911.577, 2]
            }, 
            "numeric_out_of_stock": {
                "definition": "Percentage of stores in the universe that faced Out of stock for the selected product line",
                "sample_values": [0.000047, 0.0032943, 4.4803892]
            }, 

        },
    },

    "raw_brand_loss_gain": {
        "columns": ["incr_or_decr_in_cons_of_ref_brand", "entry_to_or_lapse_from_category", "addn_or_deletion_from_repertoire", "year", "brand_net_shift",
                    "total_shift_to_or_from_ref_brand", "brand", "region"],
        "categorical_info": {
            "brand": {
                "definition": "Main brand name being evaluated for gains or losses",
                "sample_values": ["ANY Brand 1", "  ANY Brand 2 SHAMPOO", "ANY Brand 8", "ANY Brand 9", "ANY Brand 11", "ANY Brand 12", "ANY Brand 15"]
            },
            "region": {
                "definition": "Geographic or retail area (e.g., All India, Urban + Rural)",
                "sample_values": ["All India", "All India Urban", "All India Rural", "Gujarat", "Karnataka", "Maharashtra", "Rajasthan", "Up"]
            },
            "incr_or_decr_in_cons_of_ref_brand": {
                "definition": "Net gain or loss in consumption volume for the reference brand (the main brand being evaluated), based on consumer behavior changes.",
                "sample_values": [5.775, 20.528, 108.388]
            },
            "year": {
                "definition": "Year of observation",
                "sample_values": ["2022", "2023", "2024"]
            },
            "entry_to_or_lapse_from_category": {
                "definition": "Captures whether households started or stopped purchasing the overall category, not just one brand.",
                "sample_values": [0.836, -4.117, 45.607, -11.895]
            },
            "addn_or_deletion_from_repertoire": {
                "definition": "Tracks whether the reference brand was added to or removed from a household’s set of brands they typically purchase within the category.",
                "sample_values": [43.134, -150.691, 258.073, 1168.943]
            },
            "total_shift_to_or_from_ref_brand": {
                "definition": "How many households moved in or out of the reference brand (Ref. Brand), by switching from or to other brands within the same category.",
                "sample_values": [-57.663, 21.988, 398.923, -37.639]
            },
            "brand_net_shift": {
                "definition": "Sum of all above shifts: Final net Household change for the brand",
                "sample_values": [-7.918, 253.384, -745.088, 3059.762]
            },
        },
    },

    "raw_entry_erosion_consumption": {
        "columns": ["metric_new_triers", "metric_retainer", "metric_lapsers","period", "brand", "brand_filtered", "market"],
        "categorical_info": {
            "brand": {
                "definition": "Main brand category name",
                "sample_values": ["ANY Brand 1 SOAP", "ANY Brand 2 SOAP", "ANY BEAUTY SOAP", "ANY AYURVEDIC SOAP"]
            },
            "brand_filtered": {
                "definition": "Filtered brand for internal comparisons",
                "sample_values": ["Brand 1", "Brand 2", "Other"]
            },
            "market": {
                "definition": "Geographic or retail area (e.g., All India, Urban + Rural)",
                "sample_values": ["ALL India", "All India Urban", "All India Rural", "Gujarat", "Karnataka", "Maharashtra", "Rajasthan", "UP"]
            },
            "period": {
                "definition": "Reporting time period (e.g., MAT OCT'23)",
                "sample_values": ["MAT OCT'21", "MAT OCT'22", "MAT OCT'23", "MAT OCT'24"]
            },
            "metric_new_triers": {
                "definition": "Percentage of people who bought the brand for the first time (new customers)",
                "sample_values": [0.298785480050612, 1.24460963085929, 0.0213226096384299]
            },
            "metric_retainer": {
                "definition": "Percentage of buyers who bought the brand again (repeat users)",
                "sample_values": [ 0.694656476553661, 0.739106308183477, 0.978985681024135]
            },
            "metric_lapsers": {
                "definition": "Percentage of previous buyers who did not buy the brand again (lost users)",
                "sample_values": [ 0.30534352344634, 0.260893684065276, 0.303039503028793]
            },
        },
    },

    "raw_iop_hh_consumption": {
        "columns": ["market", "item", "period", "avg_cons", "avg_fop", "avg_nob", "avg_nop", "avg_poc", "avg_spent", "hh", "hh_gr_percentage",
                    "penetration_percentage", "sor_val", "sor_vol", "val", "vol", "brand_filtered"],
        "categorical_info": {
            "item": {
                "definition": "Grouped category item name",
                "sample_values": ["Product Total (000s)", "TG Base (000s)", "Universe (000s)", "[SOAP] ANY 111 - 210GM NEW", "[SOAP] ANY 26 - 69GM NEW [ANY COSMETIC SOAP]"]
            },
            "market": {
                "definition": "Geographic or retail area (e.g., All India, Urban + Rural)",
                "sample_values": ["ALL India", "All India Urban", "All India Rural", "Gujarat", "Karnataka", "Maharashtra", "Rajasthan", "UP"]
            },
            "period": {
                "definition": "Reporting time period (e.g., MAT OCT'23)",
                "sample_values": ["MAT OCT'21", "MAT OCT'22", "MAT OCT'23", "MAT OCT'24"]
            },
            "brand_filtered": {
                "definition": "Refined or grouped brand view",
                "sample_values": ["Other", "Brand 1", "Brand 2", "Brand 11", "Brand 20"]
            },
            "avg_cons": {
                "definition": "Average consumption per household",
                "sample_values": [0.89, 0.964, 305199.401, 82090]
            },
            "avg_fop": {
                "definition": "Average frequency of purchase per household",
                "sample_values": [10.296, 305199.401, 82884, 2.292]
            },
            "avg_nob": {
                "definition": "Average number of brands bought per household (brand switching behavior)",
                "sample_values": [3.401, 305199.401, 82090, 4.425]
            },
            "avg_nop": {
                "definition": "The average number of purchase occasions (shopping trips or times the product was bought) per household during the period.",
                "sample_values": [137.466, 154.088, 313490.563, 82090]
            },
            "avg_poc": {
                "definition": "the average quantity of product purchased per purchase occasion (trip) by buying households over a defined period",
                "sample_values": [26.824, 305199.401, 82884, 1.849]
            },
            "avg_spent": {
                "definition": "Average amount spent per household",
                "sample_values": [283.809, 322035.97, 290.313, 62.688]
            },
            "hh": {
                "definition": "Number of households purchasing the brand/category",
                "sample_values": [295152.1, 304027.691, 312764.221, 82090]
            },
            "hh_gr_percentage": {
                "definition": "Percentage of growth in household count over previous year",
                "sample_values": [-3.85567227693214, 1.49877741793223, 145.009542195663, -12.3495655008658]
            },
            "penetration_percentage": {
                "definition": "Percentage of total market households that bought the brand/category",
                "sample_values": [0.136648102399126, 0.0965325780570585, 0.00474113497321433]
            },
            "sor_val": {
                "definition": "Share of requirements by value (how much of total spend is on the brand)",
                "sample_values": [100, 305199.401, 322035.97, 42.774]
            },
            "sor_vol": {
                "definition": "Share of requirements by volume (how much of total quantity is from the brand)",
                "sample_values": [100, 25, 305199.401, 22.162]
            },
            "val": {
                "definition": "Total value of sales (in currency)",
                "sample_values": [83766.704, 111854.283, 330844.863]
            },
            "vol": {
                "definition": "Total quantity sold (in litres, ml, grams, etc.)",
                "sample_values": [262664.011, 305199.401, 83628]
            },
        },
    },     

    "PRI_iop_hh_consumption": {
        "columns": ["market", "item", "item_type", "pack_size", "period", "avg_cons", "avg_fop", "avg_nob", "avg_nop", "avg_poc", "avg_spent", "hh", "hh_gr_percentage",
                    "penetration_percentage", "sor_val", "sor_vol", "val", "vol", "brand_filtered"],
        "categorical_info": {
            "item": {
                "definition": "Grouped category item name",
                "sample_values": ["Product Total (000s)", "TG Base (000s)", "Universe (000s)", "[SOAP] ANY 111 - 210GM NEW", "[SOAP] ANY 26 - 69GM NEW [ANY COSMETIC SOAP]"]
            },
            "item_type": {
                "definition": "Type of the item",
                "sample_values": ["Bars", "Soap",]
            },
            "pack_size": {
                "definition": "Size of the Item in GM",
                "sample_values": [210, 69, 410, 100,]
            },
            "market": {
                "definition": "Geographic or retail area (e.g., All India, Urban + Rural)",
                "sample_values": ["ALL India", "All India Urban", "All India Rural", "Gujarat", "Karnataka", "Maharashtra", "Rajasthan", "UP"]
            },
            "period": {
                "definition": "Reporting time period (e.g., MAT OCT'23)",
                "sample_values": ["MAT OCT'21", "MAT OCT'22", "MAT OCT'23", "MAT OCT'24"]
            },
            "brand_filtered": {
                "definition": "Refined or grouped brand view",
                "sample_values": ["Other", "Brand 1", "Brand 2", "Brand 11", "Brand 20"]
            },
            "avg_cons": {
                "definition": "Average consumption per household",
                "sample_values": [0.89, 0.964, 305199.401, 82090]
            },
            "avg_fop": {
                "definition": "Average frequency of purchase per household",
                "sample_values": [10.296, 305199.401, 82884, 2.292]
            },
            "avg_nob": {
                "definition": "Average number of brands bought per household (brand switching behavior)",
                "sample_values": [3.401, 305199.401, 82090, 4.425]
            },
            "avg_nop": {
                "definition": "The average number of purchase occasions (shopping trips or times the product was bought) per household during the period.",
                "sample_values": [137.466, 154.088, 313490.563, 82090]
            },
            "avg_poc": {
                "definition": "the average quantity of product purchased per purchase occasion (trip) by buying households over a defined period",
                "sample_values": [26.824, 305199.401, 82884, 1.849]
            },
            "avg_spent": {
                "definition": "Average amount spent per household",
                "sample_values": [283.809, 322035.97, 290.313, 62.688]
            },
            "hh": {
                "definition": "Number of households purchasing the brand/category",
                "sample_values": [295152.1, 304027.691, 312764.221, 82090]
            },
            "hh_gr_percentage": {
                "definition": "Percentage of growth in household count over previous year",
                "sample_values": [-3.85567227693214, 1.49877741793223, 145.009542195663, -12.3495655008658]
            },
            "penetration_percentage": {
                "definition": "Percentage of total market households that bought the brand/category",
                "sample_values": [0.136648102399126, 0.0965325780570585, 0.00474113497321433]
            },
            "sor_val": {
                "definition": "Share of requirements by value (how much of total spend is on the brand)",
                "sample_values": [100, 305199.401, 322035.97, 42.774]
            },
            "sor_vol": {
                "definition": "Share of requirements by volume (how much of total quantity is from the brand)",
                "sample_values": [100, 25, 305199.401, 22.162]
            },
            "val": {
                "definition": "Total value of sales (in currency)",
                "sample_values": [83766.704, 111854.283, 330844.863]
            },
            "vol": {
                "definition": "Total quantity sold (in litres, ml, grams, etc.)",
                "sample_values": [262664.011, 305199.401, 83628]
            },
        },
    }                                    
}


from langchain_core.prompts import PromptTemplate  # NOT langchain.prompts
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

class Plan(BaseModel):
    steps: List[str] = Field(
        description="Step-by-step plan using available tools: get_schema_info, generate_sql, validate_sql, execute_sql, summarize_result, compare_results, clarify_ambiguous_query"
    )

import re
import openai
from typing import Dict, List

class QueryProcessor:
    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key"""
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.sql_cache = {}  # Cache dictionary for SQL queries keyed by prompt string
        
    def classify_intent(self, prompt: str) -> str:
        # Check cache first
        if prompt in self.sql_cache:
            print("Returning cached SQL query for classifying Intent.")
            return self.sql_cache[prompt]
        
        """Classify user prompt into intent categories (simple, analysis, compare)"""
        messages = [
            {
                "role": "system",
                "content": "Classify prompts into: simple (single query), analysis (with metrics), compare (between entities). Only respond with one word."
            },
            {
                "role": "user",
                "content": f"Prompt: {prompt}\nClassify intent (simple, analysis, compare):"
            }
        ]
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )
        print(response.usage, "For Intent Classification")
        # Save to cache
        self.sql_cache[prompt] = response.choices[0].message.content.strip().lower()
        return response.choices[0].message.content.strip().lower()

    def extract_entities(self, prompt: str) -> dict:
        """Extract years, brands, and pack sizes from the prompt"""
        # Extract years
        years = re.findall(r"\b(20\d{2})\b", prompt)  # Matches 2000-2099
        
        # Extract brand names (Brand followed by number/name)
        brands = re.findall(r"\b(Brand\s+\d+|Brand\s+[A-Z][a-zA-Z]*)\b", prompt, flags=re.IGNORECASE)
        
        # Extract pack sizes (numbers followed by size units)
        pack_sizes = re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:gm|gram|ml|liter|GM|ML|LITER)?\b", prompt, flags=re.IGNORECASE)
        
        # Also look for simple numbers that might be pack sizes
        if not pack_sizes:
            pack_sizes = re.findall(r"\b(\d+(?:\.\d+)?)\b", prompt)
        
        # Extract general products (fallback)
        products = re.findall(
            r"\b(?:for|vs|versus|and|compare|between)\s+([A-Z][a-zA-Z0-9\s]*)", 
            prompt,
            flags=re.IGNORECASE
        )
        
        # Combine brands and products, prioritizing brands
        all_products = []
        if brands:
            all_products.extend([brand.title() for brand in brands])  # Proper case
        if products and not brands:
            all_products.extend(products)
        
        return {
            "years": list(set(years)) if years else ["y"],
            "products": list(set(all_products)) if all_products else ["x"],
            "brands": list(set([brand.title() for brand in brands])) if brands else [],
            "pack_sizes": list(set(pack_sizes)) if pack_sizes else []
        }

    def create_anonymous_template(self, intent: str, entities: dict, original_prompt: str = '') -> str:
        """Create anonymized query template based on intent and entities"""
        prompt_lower = original_prompt.lower().strip()

        # Enhanced schema query detection
        schema_keywords = {
            "table": [
                "what tables", "list tables", "show tables", "tables in the database",
                "how many tables", "total tables", "count of tables", "table count"
            ],
            "column": [
                "list columns", "columns in", "show columns", "describe table",
                "what columns", "column names", "table structure"
            ],
            "schema": [
                "database schema", "show schema", "describe database",
                "what's the structure", "how is the database organized"
            ]
        }

        # Check for schema queries first
        for query_type, keywords in schema_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                if query_type == "table":
                    return "SCHEMA_QUERY: List all tables in the database."
                elif query_type == "column":
                    table_name = self._extract_table_name(original_prompt)
                    if table_name:
                        return f"SCHEMA_QUERY: List all columns in table {table_name}."
                    return "SCHEMA_QUERY: Please specify which table's columns to list."
                elif query_type == "schema":
                    return "SCHEMA_QUERY: Get complete database schema information."

        # Get entities with fallbacks
        years = entities.get("years", [])
        products = entities.get("products", [])
        brands = entities.get("brands", [])
        pack_sizes = entities.get("pack_sizes", [])

        y = years[0] if years else "y"
        x1 = brands[0] if brands else (products[0] if products else "x1")
        x2 = products[1] if len(products) >= 2 else "x2"
        pack_size = pack_sizes[0] if pack_sizes else "pack_size"

        # Generate appropriate template based on available entities
        if intent == "simple":
            if brands and pack_sizes:
                return f"Performance of {x1} with pack size {pack_size} in year {y}."
            elif brands:
                return f"Performance of {x1} in year {y}."
            else:
                return f"Performance of {x1} in year {y}."
        elif intent == "analysis":
            if brands and pack_sizes:
                return f"Analyze performance of {x1} with pack size {pack_size} in year {y} including key metrics (m1, m2, m3)."
            elif brands:
                return f"Analyze performance of {x1} in year {y} including key metrics (m1, m2, m3)."
            else:
                return f"Analyze performance of {x1} in year {y} including key metrics (m1, m2, m3)."
        elif intent == "compare":
            return f"Compare performance between {x1} and {x2} in year {y}."
        
        return "Unknown intent. Cannot generate template."

    def deanonymize(self, template: str, mapping: Dict[str, str]) -> str:
        """Replace placeholders with actual values"""
        for placeholder, actual in mapping.items():
            template = template.replace(placeholder, actual)
        return template
    

    def process_query(self, query: str) -> dict:
        """Complete processing pipeline for a user query"""
        # Step 1: Intent classification
        intent = self.classify_intent(query)
        
        # Step 2: Entity extraction
        entities = self.extract_entities(query)
        
        # Step 3: Template anonymization
        anonymized = self.create_anonymous_template(intent, entities, query)
        
        # Step 4: Create mapping for de-anonymization
        mapping = {}
        brands = entities.get("brands", [])
        products = entities.get("products", [])
        pack_sizes = entities.get("pack_sizes", [])
        years = entities.get("years", [])
        
        # Map brands first, then products as fallback
        if "x1" in anonymized:
            if brands:
                mapping["x1"] = brands[0]
            elif products:
                mapping["x1"] = products[0]
        
        if "x2" in anonymized and len(products) > 1:
            mapping["x2"] = products[1]
        if "x" in anonymized:
            if brands:
                mapping["x"] = brands[0]
            elif products:
                mapping["x"] = products[0]
        if "y" in anonymized and years:
            mapping["y"] = years[0]
        if "pack_size" in anonymized and pack_sizes:
            mapping["pack_size"] = pack_sizes[0]
        
        # Step 5: De-anonymize
        deanon = self.deanonymize(anonymized, mapping)

        # # Step 6: Generate SQL for master query
        # sql_query = self.generate_sql(anonymized, intent, entities)
        
        return {
            "original_query": query,
            "intent": intent,
            "entities": entities,
            "anonymized_template": anonymized,
            "mapping": mapping,
            "deanon_result": deanon,
            # "sql_query": sql_query,   # THIS IS THE KEY CHANGE
            "sql_plan": self.generate_sql_plan(intent, entities)
        }


    # def generate_sql(self, anonymized_template: str, intent: str, entities: dict) -> str:
    #     """
    #     Generate an SQL query string based on the user intent and extracted entities.
        
    #     Args:
    #         anonymized_template (str): The template-style version of the query with placeholders.
    #         intent (str): Query type: 'simple', 'analysis', 'compare', etc.
    #         entities (dict): Extracted entities like products, years, etc.
        
    #     Returns:
    #         str: The generated SQL query.
    #     """
    #     # Example: default base tables for product performance
    #     base_table = "sales_data_tbl"
        
    #     # Extract commonly used entities
    #     product = entities.get("products", [None])[0]
    #     year = entities.get("years", [None])[0]
        
    #     if intent == "simple":
    #         # Simple select query filtered by product and year
    #         sql = f"""
    #         SELECT product, SUM(sales) AS total_sales, SUM(quantity) AS total_quantity
    #         FROM {base_table}
    #         WHERE 1=1
    #         """
    #         if product:
    #             sql += f" AND product = '{product}'"
    #         if year:
    #             sql += f" AND year = {year}"
    #         sql += " GROUP BY product"
    #         return sql.strip()
        
    #     elif intent == "analysis":
    #         # Analytical query might include more aggregations and comparisons
    #         sql = f"""
    #         SELECT product, year,
    #             SUM(sales) AS total_sales,
    #             SUM(quantity) AS total_quantity,
    #             AVG(sales) AS avg_sales,
    #             -- Example YoY growth placeholder; implement as needed
    #             NULL AS yoy_growth
    #         FROM {base_table}
    #         WHERE 1=1
    #         """
    #         if product:
    #             sql += f" AND product = '{product}'"
    #         if year:
    #             sql += f" AND year = {year}"
    #         sql += """
    #         GROUP BY product, year
    #         ORDER BY year DESC
    #         """
    #         return sql.strip()
        
    #     elif intent == "compare":
    #         # Comparison between two products or years - assuming at least two entities
    #         product1 = entities.get("products", [None, None])[0]
    #         product2 = entities.get("products", [None, None])[1]
            
    #         sql = f"""
    #         SELECT product, year, SUM(sales) AS total_sales
    #         FROM {base_table}
    #         WHERE 1=1
    #         """
    #         condition_clauses = []
    #         if product1 and product2:
    #             condition_clauses.append(f"product IN ('{product1}', '{product2}')")
    #         elif product1:
    #             condition_clauses.append(f"product = '{product1}'")
    #         if year:
    #             condition_clauses.append(f"year = {year}")
    #         if condition_clauses:
    #             sql += " AND " + " AND ".join(condition_clauses)
    #         sql += """
    #         GROUP BY product, year
    #         ORDER BY product, year
    #         """
    #         return sql.strip()
        
    #     else:
    #         return "-- Unsupported intent - cannot generate SQL query"


    def generate_sql_plan(self, intent: str, entities: dict) -> List[str]:
        """Generate a step-by-step SQL execution plan"""
        steps = ["1. Verify database connection and permissions"]

        # Handle schema queries differently
        if intent.startswith("SCHEMA_QUERY"):
            steps.extend([
                "2. Query database metadata tables",
                "3. Retrieve schema information from system catalogs",
                "4. Format results for clear presentation",
                "5. Return complete schema documentation"
            ])
        elif intent == "simple":
            steps.extend([
                "2. Identify relevant tables for product performance",
                "3. Construct SELECT query with basic metrics",
                "4. Filter by year and product",
                "5. Return summarized results"
            ])
  
        elif intent == "analysis":
            steps.extend([
                "2. Identify tables containing detailed metrics",
                "3. Construct analytical query with aggregations",
                "4. Apply time-based filtering",
                "5. Include comparative metrics (YoY, MoM if available)",
                "6. Generate visualizable output"
            ])
        elif intent == "compare":
            steps.extend([
                "2. Identify comparison metrics in schema",
                "3. Build query with JOIN for product comparison",
                "4. Apply time constraints",
                "5. Calculate relative performance metrics",
                "6. Prepare side-by-side results"
            ])
        else:
            steps.append("2. Unable to generate plan - unknown intent")
        
        return steps
    
    def _extract_table_name(self, prompt: str) -> str:
        """Extract table name from column-related queries"""
        # Simple pattern matching for table names
        patterns = [
            r"columns in (?:table )?([a-zA-Z_]+)",
            r"describe (?:table )?([a-zA-Z_]+)",
            r"structure of (?:table )?([a-zA-Z_]+)"
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1)
        return ''
    

class KPISelector:
    def __init__(self):
        self.kpi_library = {
            # Sales Performance Metrics
            "Sales Growth Percentage": {
                "definition": "Total sales growth percentage",
                # "logic": "((SUM(sales_value) - SUM(sales_value)) / SUM(sales_value)) * 100", #(Sales Value TY – Sales Value LY) ÷ Sales Value LY × 100
                "logic": "((Sales Value this year – Sales Value last year) ÷ Sales Value last year) × 100)", 
                "tables": ["raw_sales_rms"],
                "keywords": ["sales", "growth", "percentage","performance"]
            },
            "Volume Growth Percentage": {
                "definition": "Total volume growth percentage",
                "logic": "((SUM(sales_vol) - SUM(sales_vol)) / SUM(sales_vol)) * 100", #(Sales Value TY – Sales Value LY) ÷ Sales Value LY × 100
                "tables": ["raw_sales_rms"],
                "keywords": ["volume", "growth", "percentage", "performance"]
            },
            "Unit Growth Percentage": {
                "definition": "Total Unit growth percentage",
                "logic": "((SUM(Unit Value for End Period) - SUM(Unit Value for Start Period)) / SUM(Unit Value for Start Period)) * 100", #(Sales Value TY – Sales Value LY) ÷ Sales Value LY × 100
                "tables": ["raw_sales_rms"],
                "keywords": ["unit", "growth", "percentage"]
            },
            "Share Change Value": {
                "definition": "Share Change Value",
                "logic": "(((Brand share Value ÷ Category Value End Period)  * 100) – ((Brand share Value ÷ Category Value Start Period) * 100))", 
                "tables": ["raw_sales_rms"],
                "keywords": ["Share", "Change", "Value", "Category"]
            },
            "Share Change Volume": {
                "definition": "Share Change Volume",
                "logic": "(((Brand share Volume ÷ Category Volume End Period)  * 100) – ((Brand share Volume ÷ Category Volume Start Period) * 100))", 
                "tables": ["raw_sales_rms"],
                "keywords": ["Share", "Change", "Volume", "Category"]
            },
            # This for IO HH Consumption.
            "Household Penetration Growth": {
                "definition": "Change in Household Penetration between this year and last year",
                "logic": "((HH / Total_Market_HH) * 100 in End Period) - ((HH / Total_Market_HH) * 100 in Start Period)",
                "tables": ["PRI_iop_hh_consumption"],
                "keywords": ["Household", "Penetration", "Growth", "Points", "Yearly Change", "Percentage"]
            },

            # This is for Entry erosion consumer data.
            "Metric Retainer": {
                "definition": "Percentage of buyers who bought the brand again (repeat users)",
                "logic": "(Repeat_Buyers / Previous_Buyers) * 100",
                "tables": ["raw_entry_erosion_consumption"],
                "keywords": ["Metric", "Retainer", "Percentage", "Repeat", "Buyers"]
            },
            "Metric Lapsers": {
                "definition": "Percentage of previous buyers who did not buy the brand again (lost users)",
                "logic": "(Lapsed_Buyers / Previous_Buyers) * 100",
                "tables": ["raw_entry_erosion_consumption"],
                "keywords": ["Metric", "Lapsers", "Percentage", "Lost", "Buyers"]
            },
            "Metric New Triers": {
                "definition": "Percentage of people who bought the brand for the first time (new customers)",
                "logic": "(New_Customers / Total_Customers) * 100",
                "tables": ["raw_entry_erosion_consumption"],
                "keywords": ["Metric", "New", "Percentage", "Triers", "Buyers"]
            },
            "Top Pack Size by Sales Volume": {
                "definition": "Lists the top-selling pack sizes by sales volume, optionally filtered by product/brand/item.",
                "logic": (
                    "SELECT basepacksize, SUM(sales_vol) AS total_volume "
                    "FROM raw_sales_rms "
                    "WHERE {filters} "
                ),
                "tables": ["raw_sales_rms"],
                "keywords": ["top", "pack size", "selling", "most", "aloe vera", "soap", "volume"]
            },
            # Add more KPI's - This is for raw_brand_loss_gain
            # "incr_or_decr_in_cons_of_ref_brand", "entry_to_or_lapse_from_category", "addn_or_deletion_from_repertoire", "year", "brand_net_shift",
            #         "total_shift_to_or_from_ref_brand", "brand", "region"
            # "Increase or Decrease in Consumption of Reference Brand": {
            #     "definition": "How many households moved in or out of the reference brand (Ref. Brand), by switching from or to other brands within the same category.",
            #     "logic": "",
            #     "tables": ["raw_brand_loss_gain"],
            #     "keywords": ["Increase", "Decrease", "Consumption", "Reference"]
            # },

#             "Revenue": {
#                 "definition": "Total sales generated by each SKU",
#                 "logic": "SUM(quantity * item-price)",
#                 "tables": ["raw_order_report_tbl"],
#                 "keywords": ["revenue", "sales", "income", "turnover"]
#             },
#             "Units Sold": {
#                 "definition": "Volume of each SKU sold",
#                 "logic": "SUM(quantity)",
#                 "tables": ["raw_order_report_tbl"],
#                 "keywords": ["units", "volume", "quantity", "how many sold"]
#             },
# #             # Profitability Metrics
#             "Gross Margin": {
#                 "definition": "(Revenue - Cost of Goods Sold) / Revenue",
#                 "logic": "(SUM(quantity * item-price) - SUM(item-price))/SUM(quantity * item-price)",
#                 "tables": ["raw_order_report_tbl"],
#                 "keywords": ["margin", "profitability", "gross profit"]
#             },
#             # Inventory Metrics
#             "Sell-Through Rate": {
#                 "definition": "(Units Sold / Initial Stock) × 100",
#                 "logic": "(SUM(sales.quantity)/inventory.initial_stock)*100",
#                 "tables": ["sales", "inventory"],
#                 "keywords": ["sell through", "inventory efficiency", "stock performance"]
#             },
#             # Pricing Metrics
#             "Average Selling Price": {
#                 "definition": "Revenue / Units Sold",
#                 "logic": "SUM(sales.amount)/SUM(sales.quantity)",
#                 "tables": ["sales"],
#                 "keywords": ["asp", "average price", "price point"]
#             },
#             "Market Share": {
#                 "definition": "Percentage of total category sales captured by the brand.",
#                 "logic": "SUM(brand_sales.amount) / SUM(category_sales.amount)",
#                 "tables": ["brand_sales", "category_sales"],
#                 "keywords": ["market share", "share", "category share"]
#             },
#             "Relative Market Share": {
#                 "definition": "Brand's market share divided by the largest competitor's market share.",
#                 "logic": "brand_share / max_competitor_share",
#                 "tables": ["brand_sales", "competitor_sales"],
#                 "keywords": ["relative market share", "rms", "share vs competitor"]
#             },
#             "Price Positioning": {
#                 "definition": "How the brand's price compares to the average market price.",
#                 "logic": "AVG(brand_price) / AVG(market_price)",
#                 "tables": ["pricing"],
#                 "keywords": ["price positioning", "price index", "relative price"]
#             },
#             "Average Selling Price": {
#                 "definition": "Revenue / Units Sold",
#                 "logic": "SUM(sales.amount)/SUM(sales.quantity)",
#                 "tables": ["sales"],
#                 "keywords": ["asp", "average price", "price point"]
#     },
#             "Brand Awareness": {
#                 "definition": "Extent to which customers are able to recall or recognize a brand.",
#                 "logic": "AVG(surveys.brand_awareness_score)",
#                 "tables": ["surveys"],
#                 "keywords": ["brand recall", "awareness", "brand recognition"]
#     },
#              "Net Promoter Score": {
#                 "definition": "Measures customer loyalty and satisfaction based on how likely customers are to recommend.",
#                 "logic": "AVG(surveys.nps_score)",
#                 "tables": ["surveys"],
#                 "keywords": ["nps", "net promoter", "loyalty"]
#     }


        }
        
#         self.brand_manager_priority_kpis = [
#             "Revenue", "Gross Margin"
# #             , "Market Share", 
# #             "Brand Awareness", "Net Promoter Score"
#         ]
        self.brand_manager_priority_kpis = [
        "Sales Growth Percentage", "Volume Growth Percentage", "Unit Growth Percentage",
        "Share Change Value", "Share Change Volume", "Household Penetration Growth",
        "Brand Awareness", "Net Promoter Score", "Top Pack Size by Sales Volume",
        ]

    def select_relevant_kpis(self, prompt: str, intent: str) -> dict:
        """Select most relevant KPIs based on prompt and intent"""
        prompt_lower = prompt.lower()
        relevant_kpis = {}
        
        # First check for explicit KPI mentions
        for kpi, details in self.kpi_library.items():
            if any(keyword in prompt_lower for keyword in details["keywords"]):
                relevant_kpis[kpi] = details
        # If no explicit matches, use intent-based selection
        if not relevant_kpis:
            if intent == "analysis":
                # For analysis queries, use brand manager priority KPIs
                for kpi in self.brand_manager_priority_kpis:
                    relevant_kpis[kpi] = self.kpi_library[kpi]
            elif intent == "compare":
                # For comparisons focus on relative metrics
                comparison_kpis = ["Market Share", "Relative Market Share", "Price Positioning"]
                for kpi in comparison_kpis:
                    if kpi in self.kpi_library:
                        relevant_kpis[kpi] = self.kpi_library[kpi]
        
        return relevant_kpis

    def generate_kpi_explanation(self, kpi_name: str) -> str:
        """Generate human-readable explanation of KPI calculation"""
        kpi = self.kpi_library.get(kpi_name)
        if not kpi:
            return f"KPI '{kpi_name}' not found in library."
        
        return (
            f"KPI: {kpi_name}\n"
            f"Definition: {kpi['definition']}\n"
            f"Calculation: {kpi['logic']}\n"
            f"Required Tables: {', '.join(kpi['tables'])}"
        )
    
from typing import List, Union
from typing import Dict, List  # Ensure proper typing imports


class SmartQueryEngine:
    def __init__(self, client, schema_json, db_config, kpi_dict):
        """
        client: LLM client instance
        intent: e.g., "simple", "analysis"
        schema_json: schema metadata dictionary (table -> columns, categorical_info)
        db_config: database connection config dictionary for psycopg2
        kpi_dict: dictionary of KPI definitions (see previous docs)
        """
        self.client = client
        # self.intent = intent
        self.schema_json = schema_json
        self.db_config = db_config
        self.kpi_dict = kpi_dict
        self.validated_queries = []
        self.sql_cache = {}  # Cache dictionary for SQL queries keyed by prompt string

    def clean_sql(self, sql: str) -> str:
        return sql.replace('\n', ' ').replace('\t', ' ').strip()

    def get_years_from_periods(self, period_values):
        years = set()
        for p in period_values:
            match = re.search(r"'(\d{2})", p)
            if match:
                year_suffix = match.group(1)
                year_full = int('20' + year_suffix)
                years.add(year_full)
        return sorted(list(years))

    def get_periods_for_year(self, period_values, year):
        yy = str(year)[-2:]
        return [p for p in period_values if f"'{yy}" in p]

    def pick_relevant_table(self, prompt: str):
        prompt_l = prompt.lower()
        print(prompt_l)

        # Priority fallback checks first
        if any(k in prompt_l for k in ["household", "penetration", "hh"]):
            return 'PRI_iop_hh_consumption'

        if any(k in prompt_l for k in ["brand loss", "gain", "repertoire", "brand_net_shift"]):
            return "raw_brand_loss_gain"

        if any(k in prompt_l for k in ["triers", "retainer", "lapsers"]):
            return "raw_entry_erosion_consumption"

        # Then check KPIs keywords
        for kpi_name, kpi_data in self.kpi_dict.items():
            keywords = kpi_data.get('keywords', [])
            for keyword in keywords:
                if keyword.lower() in prompt_l:
                    tables = kpi_data.get("tables", [])
                    if tables:
                        return tables[0]
        # Default fallback
        return "raw_sales_rms"

    def build_table_context(self, table_name):
        table_info = self.schema_json[table_name]
        columns = table_info["columns"]
        ci = table_info.get("categorical_info", {})
        context = f"Table: {table_name}\nColumns: {', '.join(columns)}\n"
        for field, meta in ci.items():
            allowed = meta.get("sample_values", [])
            if allowed:
                truncated = allowed[:8]
                allowed_str = ', '.join(map(str, truncated))
                if len(allowed) > 8:
                    allowed_str += ", ..."
                context += f"{field}: Allowed values: {allowed_str}\n"
        return context

    def find_best_kpi_for_prompt(self, prompt: str):
        """
        Returns:
        tuple: (table_name, (kpi_name, kpi_data)) if match found, else (None, (None, None))
        """
        prompt_l = prompt.lower()
        best_match = None
        best_score = 0
        best_table = None
        for kpi_name, kpi_data in self.kpi_dict.items():
            keywords = [kw.lower() for kw in kpi_data.get('keywords', [])]
            score = sum(prompt_l.count(kw) for kw in keywords)
            if score > best_score:
                best_score = score
                best_match = (kpi_name, kpi_data)
                best_table = kpi_data.get('tables', [None])[0]
        if best_match:
            return best_table, best_match
        return None, (None, None)

    def preprocess_prompt(self, prompt: str) -> str:
        # Collect all known period values from your schema metadata
        all_periods = []
        for table_info in self.schema_json.values():
            time_values = table_info.get("categorical_info", {}).get("period", {}).get("sample_values", [])
            all_periods.extend(time_values)
        all_periods = sorted(set(all_periods))

        if not all_periods:
            return prompt  # No periods available to map

        # Map common phrases to latest periods
        phrase_to_period = {
            r'\bthe last year\b': all_periods[-1],
            r'\blast year\b': all_periods[-1],
            r'\bprevious year\b': all_periods[-2] if len(all_periods) > 1 else all_periods[-1],
            r'\bthis year\b': all_periods[-1],
            r'\btwo years ago\b': all_periods[-3] if len(all_periods) > 2 else all_periods[-1],
            r'\bthree years ago\b': all_periods[-4] if len(all_periods) > 3 else all_periods[-1],
        }

        # Replace phrases with exact period strings wrapped in single quotes
        prompt_out = prompt
        for phrase, period_val in phrase_to_period.items():
            if period_val:  # Make sure period exists
                # Escape single quotes for SQL compliance
                escaped_val = period_val.replace("'", "''")
                prompt_out = re.sub(phrase, f"'{escaped_val}'", prompt_out, flags=re.IGNORECASE)

        # Optional: Replace any standalone 'y' with latest period
        latest_period = all_periods[-1]
        prompt_out = re.sub(r'\by\b', f"'{latest_period}'", prompt_out, flags=re.IGNORECASE)

        return prompt_out
    
    # Extra Method to extract and normalize periods from prompt using schema info:

    def extract_period_entities(self, prompt: str):
        """Extract known period terms and map to actual schema periods."""
        # Extract raw years and phrases from prompt — can be enhanced with NLP
        time_phrases = {
            'last year': None,
            'previous year': None,
            'the last year': None,
            'this year': None,
            'two years ago': None,
            'three years ago': None
        }

        # Gather all periods from schema for mapping
        periods = []
        for table_info in self.schema_json.values():
            ci = table_info.get('categorical_info', {})
            table_periods = ci.get('period', {}).get('sample_values', [])
            periods.extend(table_periods)
        periods = sorted(set(periods))  # sorted ascending by date

        if not periods:
            return time_phrases

        # Map phrases to concrete period tokens (most recent last)
        def safe_period(i):
            if i < 0 or i >= len(periods):
                return None
            return periods[i]

        time_phrases['the last year'] = safe_period(-1)
        time_phrases['last year'] = safe_period(-2) or safe_period(-1)
        time_phrases['previous year'] = safe_period(-2) or safe_period(-1)
        time_phrases['this year'] = safe_period(-1)
        time_phrases['two years ago'] = safe_period(-3) or safe_period(-2) or safe_period(-1)
        time_phrases['three years ago'] = safe_period(-4) or safe_period(-3) or safe_period(-2) or safe_period(-1)

        # Detect what phrases appear in prompt
        found_phrases = {}
        for phrase in time_phrases.keys():
            if re.search(rf'\b{phrase}\b', prompt, flags=re.IGNORECASE):
                found_phrases[phrase] = time_phrases[phrase]

        return found_phrases


    # def preprocess_prompt(self, prompt: str, table_info: dict) -> str:
    #     ci = table_info.get('categorical_info', {})
    #     period_values = ci.get('period', {}).get('sample_values', [])
    #     years = self.get_years_from_periods(period_values)
    #     if not years:
    #         return prompt
    #     max_year = max(years)
    #     replacements = {
    #         r'\blast year\b': str(max_year - 1),
    #         r'\bprevious year\b': str(max_year - 1),
    #         r'\byear before last year\b': str(max_year - 2),
    #         r'\btwo years ago\b': str(max_year - 2),
    #         r'\bthree years ago\b': str(max_year - 3),
    #         r'\bthis year\b': str(max_year),
    #     }
    #     prompt_out = prompt
    #     for pattern, replacement_year in replacements.items():
    #         prompt_out = re.sub(pattern, replacement_year, prompt_out, flags=re.IGNORECASE)
    #     for year in sorted(years, reverse=True):
    #         period_list = self.get_periods_for_year(period_values, year)
    #         if not period_list:
    #             continue
    #         escaped_periods = [f"'{self.escape_sql_single_quotes(p)}'" for p in period_list]
    #         period_in_clause = f"period IN ({', '.join(escaped_periods)})"
    #         prompt_out = re.sub(
    #             rf'\b{year}\b',
    #             period_in_clause,
    #             prompt_out
    #         )
    #     return prompt_out

    def escape_sql_single_quotes(self, s):
        return s.replace("'", "''")

    def inject_kpi_logic_into_prompt(self, kpi_name, kpi_data, table_info):
        logic = kpi_data.get("logic", "")
        period_col = "period"
        ci = table_info.get('categorical_info', {})
        period_values = ci.get('period', {}).get('sample_values', [])
        years = self.get_years_from_periods(period_values)
        start_period, end_period = None, None
        if len(years) >= 2:
            end_year = years[-1]
            start_year = years[-2]
            end_periods = self.get_periods_for_year(period_values, end_year)
            start_periods = self.get_periods_for_year(period_values, start_year)
            if end_periods:
                end_period = end_periods[0]
            if start_periods:
                start_period = start_periods[0]
        if logic and '{start_period}' in logic and '{end_period}' in logic:
            if start_period and end_period:
                start_period_escaped = self.escape_sql_single_quotes(start_period)
                end_period_escaped = self.escape_sql_single_quotes(end_period)
                logic_filled = logic.format(
                    period_col=period_col,
                    start_period=start_period_escaped,
                    end_period=end_period_escaped)
            else:
                logic_filled = logic.format(
                    period_col=period_col,
                    start_period="'START_PERIOD'",
                    end_period="'END_PERIOD'")
        else:
            logic_filled = logic
        header = f"KPI: {kpi_name}\nDefinition: {kpi_data.get('definition','')}\nLogic:\n{logic_filled}\n"
        header += f"- Use column '{period_col}' with allowed values: {', '.join(period_values[:8])}"
        if len(period_values) > 8:
            header += ", ..."
        header += "\nIMPORTANT: Use this exact SQL logic for the KPI. Never invent formulas or columns.\n"
        return header

    def format_schema(self):
        schema_lines = []
        for table, table_info in self.schema_json.items():
            # Handle both old format (list) and new format (dict)
            if isinstance(table_info, list):
                # Old format compatibility
                line = f"{table} ({', '.join(table_info)})"
                schema_lines.append(line)
            elif isinstance(table_info, dict):
                # New enhanced format
                columns = table_info.get("columns", [])
                line = f"{table} ({', '.join(columns)})"
                schema_lines.append(line)
                
                # Add categorical information if available
                categorical_info = table_info.get("categorical_info", {})
                if categorical_info:
                    schema_lines.append(f"\n{table} - Categorical Fields:")
                    for field, info in categorical_info.items():
                        definition = info.get("definition", "")
                        sample_values = info.get("sample_values", [])
                        if sample_values:
                            # Limit sample values to avoid overwhelming the prompt
                            sample_display = sample_values[:10]  # Show first 10 values
                            if len(sample_values) > 10:
                                sample_display.append("...")
                            schema_lines.append(f"  - {field}: {definition}")
                            schema_lines.append(f"    Sample values: {', '.join(map(str, sample_display))}")
                
                # Add metric definitions if available
                metric_definitions = table_info.get("metric_definitions", {})
                if metric_definitions:
                    schema_lines.append(f"\n{table} - Metric Definitions:")
                    for metric, definition in metric_definitions.items():
                        schema_lines.append(f"  - {metric}: {definition}")
                        
        return "\n".join(schema_lines)
    
    # Updated on 20/08/2025:
    def extract_relevant_table(self, prompt: str):
    # Leverage your already-existing pick_relevant_table logic
        table = self.pick_relevant_table(prompt)
        return table

    def extract_relevant_columns(self, table_name: str, prompt: str):
        # Returns just the subset of columns in the table that are likely referenced in the prompt/KPI
        if table_name not in self.schema_json:
            return []
        all_cols = self.schema_json[table_name].get("columns", [])
        prompt_l = prompt.lower()
        # Use simple substring matching for now; advanced: use entity/KPI extraction heuristics
        relevant_cols = [col for col in all_cols if col.lower() in prompt_l]
        # Always include key columns for the table (id, period, brand, value, etc.) if unclear
        essential_cols = [col for col in ['brand', 'period', 'sales_value', 'category', 'basepacksize'] if col in all_cols]
        for col in essential_cols:
            if col not in relevant_cols:
                relevant_cols.append(col)
        # Limit to, say, first 8 for brevity
        return list(dict.fromkeys(relevant_cols))[:8]

    def extract_relevant_kpi(self, prompt: str):
        # Uses your find_best_kpi_for_prompt logic
        _, (kpi_name, kpi_data) = self.find_best_kpi_for_prompt(prompt)
        return kpi_name, kpi_data

    # Gather Schema related to table.
    def fetch_table_schema(self, table_name):
        query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (table_name,))
                    columns = cur.fetchall()
            schema_description = f'Table: "{table_name}"\nColumns:\n'
            schema_description += "\n".join([f"- {col} ({dtype})" for col, dtype in columns])
            return schema_description
        except Exception as e:
            print(f"Failed to fetch schema for {table_name}: {e}")
            return ""

    def generate_simple_sql(self, prompt: str) -> str:

        # Check cache first
        if prompt in self.sql_cache:
            print("Returning cached SQL query for generating SQL.")
            return self.sql_cache[prompt]
        
        # 1. Is it a meta/schema question? Answer directly.
        # schema_str = self.format_schema() if hasattr(self, "format_schema") else ""
        metadata_schema = """
        information_schema.tables (table_name, table_schema, table_type)
        information_schema.columns (table_name, column_name, data_type)
        """
        # 1. Handle DB metadata query (detected via keywords)
        prompt_l = prompt.lower()
        meta_keywords = [
            "how many tables", "number of tables", "table names", "list tables", "show tables",
            "information_schema", "describe table", "desc ", "list columns", "column names"
        ]
        db_name = os.getenv('DB_NAME') # Make sure this is properly set
        if any(kw in prompt_l for kw in meta_keywords):
            # inject metadata schema, let LLM generate raw metadata SQL
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a SQL generator (PostgreSQL). 
                    You can access business tables (see below) and information_schema.tables, information_schema.columns (shown below).
                    {metadata_schema}
                    If the user asks about tables/columns/schema, use queries on information_schema.tables or information_schema.columns. 
                    Return ONLY a valid SQL query beginning with SELECT/WITH, or ERROR. No extra explanation.
                    Database name: {db_name}"""
                },
                {"role": "user", "content": prompt}
            ]
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0
                )
                print(response.usage, "For Simple Questions")
                sql = response.choices[0].message.content.strip()
                # Save to cache
                self.sql_cache[prompt] = sql

            except Exception as e:
                print(f"⚠️ Error during LLM call: {e}")
                return "ERROR"
            if sql.upper().startswith(('SELECT', 'WITH')) or sql == 'ERROR':
                return sql
            return "ERROR"

         # 2. KPI/data analysis path: ONLY send required info
        table = self.extract_relevant_table(prompt)
        kpi_name, kpi_data = self.extract_relevant_kpi(prompt)
        if not table or not kpi_data or not kpi_name:
            return "ERROR"

        cols = self.extract_relevant_columns(table, prompt)
        cols_str = ", ".join(cols)
        metric_section = (
            f"KPI: {kpi_name}\nDefinition: {kpi_data.get('definition','')}\nLogic: {kpi_data.get('logic','')}"
        )

        # You may want to include allowed values for 1-2 key categorical fields only (optional).
        table_info = self.schema_json[table]
        cat_info = table_info.get("categorical_info", {})
        cat_section = ""
        for field in cat_info:
            if field in cols:
                vals = cat_info[field].get("sample_values", [])[:5]
                if vals:
                    cat_section += f"\n{field} allowed: {', '.join(map(str, vals))}"
        # Preprocess prompt to expand time aliases, e.g. 'last year'
        prompt_fixed = self.preprocess_prompt(prompt)
        # print(cols_str)
        # print(cat_section)

        # Fetch schema dynamically from DB
        table_schema = self.fetch_table_schema(table)
        print(table_schema, "-------- Table Schema --------")
        print(metric_section, "-------- Metric Section - KPI --------")

        system_prompt = (
            f'You are a Postgres SQL expert. The main business table is: "{table}".\n'
            # f"Columns: {cols_str}\n"
            # f"{cat_section}\n"
            f"Table Schema: {table_schema}"
            f"{metric_section}\n"
            """ *** RULES ***
            - Use only the table and columns above (no schema prefixes). If a table starts with a capital letter, wrap its name in double quotes. 
            - For dates, 'year', 'month', 'week', 'day' are INTEGER; 'period' uses values like 'MAT OCT''24'.
            - 'last year' = latest full year from above.
            - Prevent division by zero (use NULLIF or CASE).
            - Whenever there is a string comparision always perform the comparision in the same case(lower or upper) in the SQL query.
            - Match the condition strings given by the users to the nearest possible value in the conditional columns. For Example: If user is asking about 'aloevera' the user intends to enquire about 'aloe vera' form the fragrance_ingredients column.
            - Use 'period' for time filtering, with allowed values as shown.
            - No explanations or extra text. Output only a valid SELECT/WITH SQL or 'ERROR'.
            - For unsupported metrics, output 'ERROR'. """
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_fixed}
                ],
                temperature=0
            )
            sql = response.choices[0].message.content.strip()
            # Save to cache
            self.sql_cache[prompt] = sql
            print(response.usage, "For Analysis Questions")  # Optional: see tokens used
            print("\n=== LLM RAW SQL OUTPUT ===\n", sql, "\n==========================")
        except Exception as e:
            print(f"⚠️ Error during LLM call: {e}")
            return "ERROR"

        if sql.upper().startswith(("SELECT", "WITH")) or sql.strip() == "ERROR":
            return sql
        return "ERROR"
    
        # Till Here "Giving only relevant Schema, kpi's to llm"

        # # 2. KPI-based logic
        # table_name, (kpi_name, kpi_data) = self.find_best_kpi_for_prompt(prompt)
        # if table_name is None or kpi_name is None:
        #     table_name = self.pick_relevant_table(prompt)
        #     kpi_name, kpi_data = self.find_best_kpi_for_prompt(prompt, table_name)
        #     if kpi_name is None:
        #         # No KPI match, fallback: let LLM try a generic "performance" query,
        #         # or return ERROR for unsupported/irrelevant prompt.
        #         return "ERROR"

        # table_info = self.schema_json[table_name]
        # prompt_fixed = self.preprocess_prompt(prompt, table_info)
        # table_context = self.build_table_context(table_name)
        # metric_section = ""
        # if kpi_data:
        #     metric_section = self.inject_kpi_logic_into_prompt(kpi_name, kpi_data, table_info)
        # db_name = os.getenv('DB_NAME')

        # system_prompt = (
        #     f"You are a PostgreSQL expert SQL generator.\n"
        #     f"Use ONLY this table and exact columns/allowed values:\n"
        #     f"{table_context}\n"
        #     f"{metric_section}\n"
        #     "**********IMPORTANT RULES**********\n"
        #     "1. Use only the table shown above (no schema prefixes) and if Table name starts with Captial letters place it in double inverted commas\n"
        #     "2. For date/time filtering, 'year', 'month', 'week', 'day' are INTEGER columns and Period column examples are 'MAT OCT''22', 'MAT OCT''23'\n"
        #     "3. For 'last year' or similar, use the latest complete year period as shown above.\n"
        #     f"4. Use {schema_json} for table names, column definitions and sample values. Use these values properly because you generated Sql query with column name as 'pack_sizeperiod', but the 'pack_size' and 'period' are two different columns with different meaning\n"
        #     "5. When the query involves division, ensure that division by zero is safely handled by either using NULLIF() or a CASE statement to avoid errors."
        #     "6. Use 'period' for time filtering, with allowed values from sample list above.\n"
        #     "7. Do NOT include any explanation, comment, or extra text.\n"
        #     "8. PRI_iop_hh_consumption provides insights at item, brand_filtered, and period level. Always use these three columns in SELECT."
        #     "9. Always try understanding prompt breifly and give the sql query with repect to theprompt."
        #     f"10. Use {queries_data} for the sample prompts and sql queries."
        #     "11. For unsupported metrics, reply ONLY with 'ERROR'.\n"
        #     "12. Only return a valid SQL query or 'ERROR'.\n"
        # ) 

        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": prompt_fixed}
        # ]
        # try:
        #     response = self.client.chat.completions.create(
        #         model="gpt-4",
        #         messages=messages,
        #         temperature=0
        #     )
        #     print(response.usage, "For Analysis Questions")
        #     sql = response.choices[0].message.content.strip()
        #     print("\n=== LLM RAW SQL OUTPUT ===\n", sql, "\n==========================")
        # except Exception as e:
        #     print(f"⚠️ Error during LLM call: {e}")
        #     return "ERROR"

        # if sql.upper().startswith(("SELECT", "WITH")) or sql.strip() == "ERROR":
        #     return sql
        # return "ERROR"

    def extract_kpis_and_prompts(self, prompt: str) -> list:      
        messages = [
            {
                "role": "system",
                "content": """You are a data assistant. From the given user prompt:
                1. Extract relevant KPIs (e.g., Sales Growth Percentage, Volume Growth Percentage, Share Change Value, Household Penetration Growth)
                2. Create 3–5 follow-up user prompts, each focusing on one KPI or a time range for detailed analysis.
                3. Always provide KPIs at yearly level only.
                4. Maintain context from previous prompts such as brand, category, pack type.
                5. Do not repeat previously asked prompts.
                Respond in this format:
                KPIs: [kpi1, kpi2]
                Prompts:
                - prompt 1
                - prompt 2
                ..."""
            },
            {"role": "user", "content": f"Prompt: {prompt}"}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )
        print(response.usage, "For Drill Down Question generation.")
        return response.choices[0].message.content.strip()

    def handle_prompt(self, user_prompt: str, intent=None):
        # intent = self.intent
        print(f"Intent: {intent}")
        if intent == "simple":
            # return self.generate_simple_sql(user_prompt)
            return "It Has generated simple query already"
        elif intent == "analysis":
            breakdown_text = self.extract_kpis_and_prompts(user_prompt)
            prompts = []
            capturing = False
            for line in breakdown_text.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    prompts.append(line[2:])
                    capturing = True
                elif capturing and line:
                    prompts.append(line)
            sql_outputs = []
            for sub_prompt in prompts:
                sql = self.generate_simple_sql(sub_prompt)
                sql = self.clean_sql(sql)
                sql_outputs.append({"prompt": sub_prompt, "sql": sql})
            return {
                "original_prompt": user_prompt,
                "intent": intent,
                "sub_prompts_sql": sql_outputs
            }
        else:
            return f"Intent '{intent}' not yet supported."

    def validate_queries_with_explain(self, sql_outputs: list) -> dict:
        successful = []
        failed = []
        for item in sql_outputs:
            prompt = item.get("prompt")
            sql = item.get("sql", "").strip()
            if sql == "ERROR":
                failed.append({"prompt": prompt, "sql": sql, "error": "LLM failed to generate valid SQL"})
                continue
            if not (sql.upper().startswith("SELECT") or sql.upper().startswith("WITH")):
                failed.append({"prompt": prompt, "sql": sql, "error": "Not a SELECT or WITH statement"})
                continue
            try:
                with psycopg2.connect(**self.db_config) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(f"EXPLAIN {self.clean_sql(sql)}")
                successful.append({"prompt": prompt, "sql": sql})
            except Exception as e:
                failed.append({"prompt": prompt, "sql": sql, "error": str(e)})
        self.validated_queries = successful
        return {"successful": successful, "failed": failed}

    def fix_failed_queries_with_llm(self, failed_queries):
        fixed_results = []
        if not hasattr(self, 'validated_queries'):
            self.validated_queries = []
        for item in failed_queries:
            prompt = item.get("prompt")
            sql = item.get("sql")
            table_name = self.pick_relevant_table(prompt)
            table_context = self.build_table_context(table_name)
            explain_sql = f"EXPLAIN {sql}"
            error_msg = item.get("error", "")
            try:
                with psycopg2.connect(**self.db_config) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(explain_sql)
                continue  # No fix needed
            except Exception as e:
                error_msg = str(e)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a senior SQL developer. You are given a user prompt, a SQL query, and the EXPLAIN error. "
                        "Suggest a corrected SQL query fixing the error. Use ONLY the table and columns below.\n"
                        + table_context +
                        "\nReturn only the corrected SQL, no explanations."
                    )
                },
                {
                    "role": "user",
                    "content": f"User prompt: {prompt}\nOriginal SQL: {sql}\nEXPLAIN Error: {error_msg}"
                }
            ]
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0
            )
            print(response.usage, "For Fixing LLM Failed LLM Queries")
            suggestion = response.choices[0].message.content.strip()
            explain_success = False
            try:
                with psycopg2.connect(**self.db_config) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(f"EXPLAIN {suggestion}")
                explain_success = True
            except Exception:
                explain_success = False
            fixed_results.append({
                "prompt": prompt,
                "original_sql": sql,
                "error": error_msg,
                "llm_suggestion": suggestion,
                "explain_success": explain_success
            })
            if explain_success:
                self.validated_queries.append({"prompt": prompt, "sql": suggestion})
        return {"fixed_results": fixed_results, "validated_queries": self.validated_queries}

    def execute_sql(self, sql: str) -> pd.DataFrame:
        try:
            with psycopg2.connect(**self.db_config) as conn:
                df = pd.read_sql(sql, conn)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to execute SQL: {e}")


if __name__ == "__main__":
    # Initialize with your OpenAI API key
    # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    processor = QueryProcessor(openai_api_key=os.getenv("OPENAI_API_KEY"))
    kpi_selector = KPISelector()
    
    # test_query = "What was the conversion rate of sku1 in 2025?"    
    # test_query = "How many number of tables are in out database?"    
    # test_query = "can you provide the performance of Brand 1 and pack size of 180"    
    # test_query = "How Many people tried Brand 1 for the first time"    
    # test_query = "can you provide the share performance of Brand 1 and pack size of 180"    
    # test_query = "can you provide the metric performance of Brand 1"      
    # test_query = "What is the Sales Growth Percentage of Brand 1 with a pack size of 180 for the last year?"    
    test_query = "What is the Household Penetration Growth of Brand 1 for the last year?"    
    # test_query = "Which soap pack sizes are selling the most, especially aloe vera soaps?"    
    # test_query = "What is the Sales Growth Percentage of Brand 2 with aloevera fragrance for the last year?"  
     
    # test_query = "What is the Performance of all the brands with aloevera fragrance for the last year?"    
    # test_query = "What is the Performance of all the brands with greentea fragrance for the last year?"    

    # test_query = "How many units of all brands with aloevera fragrance were sold in 2024?"     
    # test_query = "What products have shown the highest performance by revenue"    
    # test_query = "Give me the table names from the database?"    
    # test_query = "How do soap pack size preferences differ by region, especially for aloe vera soaps?"    
    # test_query = "What is the Household Penetration Growth for Brand 11 with a pack size of 100 over the past year"    
    # test_query = "Give the Household Penetration Growth for brands and items with in the last year?"    
    # test_query = "How has the Household Penetration Growth changed for brands with aloevera fragrance in the last year?"   


    # How has the Household Penetration Growth changed for brands with aloevera fragrance in the last year?
    # What is the Performance of all the brand 1 bottle versus all brands combined? 
    
    # test_query = "FMCG quarterly growth in urban market?"

    result = processor.process_query(test_query)
    print(result)
    intent = result['intent']

    # master_sql_query = result['sql_query']
    DB_CONFIG = {
            "host": os.getenv("DB_HOST"),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": os.getenv("DB_PORT")
        }
    smart_engine = SmartQueryEngine(
        client=client,
        # intent=intent,
        schema_json=schema_json,
        db_config=DB_CONFIG,  # This must be properly configured
        kpi_dict= kpi_dict
    )
    
    # Use the original prompt or de-anonymized prompt for SQL generation
    sql = smart_engine.generate_simple_sql(result["original_query"])
    print(sql)

    # Validate/fix if desired
#     validation_results = smart_engine.validate_queries_with_explain([{"prompt": result["original_query"], "sql": sql}])
#     print(validation_results)
#     if validation_results["failed"]:
#         fix_results = smart_engine.fix_failed_queries_with_llm(validation_results["failed"])
#         print(fix_results)
#         print(fix_results['fixed_results'][0]['llm_suggestion'])
#         fixed_sql = fix_results['fixed_results'][0]['llm_suggestion']
#     print("\n=== Query Processing Results ===")
#     print(f"Original Query: {result['original_query']}")
#     print(f"Intent: {result['intent']}")
#     print(f"Entities: {result['entities']}")
#     print("\n=== SQL Execution Plan ===")
#     for step in result["sql_plan"]:
#         print(f"- {step}")
    
#     # Get and display relevant KPIs
#     selected = kpi_selector.select_relevant_kpis(
#         prompt=result['original_query'], 
#         intent=result['intent']  # Fixed variable name from intnt to result['intent']
#     )
#     # print(selected, "Debug")
#     if selected:
#         print("\n=== Recommended KPIs ===")
#         for kpi, details in selected.items():
#             print(f"\n{kpi}:")
#             print(f"  Definition: {details['definition']}")
#             print(f"  Calculation: {details['logic']}")
#             print(f"  Tables: {', '.join(details['tables'])}")
    

#     DB_CONFIG = {
#             "host": os.getenv("DB_HOST"),
#             "database": os.getenv("DB_NAME"),
#             "user": os.getenv("DB_USER"),
#             "password": os.getenv("DB_PASSWORD"),
#             "port": os.getenv("DB_PORT")
#         }

# #     engine = SmartQueryEngine(client=client, intent=intent, schema_json=schema_json, db_config=DB_CONFIG)
#     engine = SmartQueryEngine(
#         client=client,
#         # intent=intent,
#         schema_json=schema_json,
#         db_config=DB_CONFIG,  # This must be properly configured
#         kpi_dict= kpi_dict
#     )
#     # print(engine)
#     result = engine.handle_prompt(test_query, intent=intent)
#     print('-----------')
#     print(result, "Debug")
#     print('-----------')


#     # Type-safe checking
#     if hasattr(result, "get") and "sub_prompts_sql" in result:
#         validation = engine.validate_queries_with_explain(result["sub_prompts_sql"])
#         print("validation", validation)
        
#         print("\nSuccessful queries:")
#         for success in validation["successful"]:
#             print(f"- {success['prompt']}")
#             print(f"  SQL: {success['sql'][:100]}...")  # Show first 100 chars          

#         print("\nFailed queries:")
#         for failure in validation["failed"]:
#             print(f"- {failure['prompt']}")
#             print(f"  Error: {failure['error']}") 

#         # fix_results = engine.fix_failed_queries_with_llm(validation["failed"])
#         # print(fix_results)
#     else:
#         print("Unexpected result format:", result)



# Testing each method to get relavant table Info, KPIs and Columns:
    # relevant_table = smart_engine.extract_relevant_table(test_query)    
    # relevant_kpi = smart_engine.extract_relevant_kpi(test_query)    
    # relevant_columns = smart_engine.extract_relevant_columns(relevant_table, test_query)   

    # print(relevant_table) 
    # print(relevant_kpi) 
    # print(relevant_columns)   

    # prompt_fixed = smart_engine.preprocess_prompt(test_query)
    # print(prompt_fixed)

    # schema_table = smart_engine.fetch_table_schema(relevant_table)
    # print(schema_table)