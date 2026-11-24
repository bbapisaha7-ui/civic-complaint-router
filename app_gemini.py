import os
import json
from datetime import date

import streamlit as st
from google import genai
from google.genai import types

# Models are tried in this order until one works. Edit if Google renames them.
MODELS = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]

# ---------- Department knowledge (keeps the AI grounded) ----------
# Edit this list for YOUR city/state. Names differ everywhere.
DEPARTMENTS = {
    "Streetlight / Electricity pole": "Municipal Corporation (street lighting wing) or State Electricity Distribution Company",
    "Water supply / Pipe leak": "Municipal Water Supply Department / Public Health Engineering Department (PHED)",
    "Garbage / Sanitation": "Municipal Corporation - Solid Waste Management / Sanitation Department",
    "Road damage / Potholes": "Public Works Department (PWD) or Municipal Corporation Roads Wing",
    "Drain / Sewage blockage": "Municipal Corporation - Drainage and Sewerage Department",
    "Stray animals": "Municipal Corporation - Veterinary / Animal Control Wing",
    "Traffic / Illegal parking": "Traffic Police",
    "Public health hazard (mosquito, dirty water)": "Municipal Health Department / Health Officer",
    "Other": "District Collector / Municipal Commissioner's office",
}

SYSTEM_PROMPT = f"""You are a civic complaint assistant for citizens in India.
Given a problem description (and optionally a photo), you must:
1. Classify the problem into exactly one category from this list:
{json.dumps(list(DEPARTMENTS.keys()), indent=2)}
2. Choose the responsible department using this mapping:
{json.dumps(DEPARTMENTS, indent=2)}
3. Rate urgency: Low, Medium, High (High = safety risk such as exposed wires, open manholes, sewage on road).
4. Write a polite, formal complaint letter addressed to the head of that department.
   Use the placeholders exactly as provided by the user. Do not invent names, phone numbers or reference numbers.
5. Write the letter in the language requested by the user.

Reply with ONLY valid JSON, no markdown, in this shape:
{{
  "category": "...",
  "department": "...",
  "urgency": "Low|Medium|High",
  "reason": "one short sentence explaining the classification",
  "subject_line": "...",
  "letter": "full letter text",
  "next_steps": ["step 1", "step 2", "step 3"]
}}
If a photo is given, base your answer on what is visible plus the text. If the photo is unclear, say so in "reason"."""


def get_client():
    key = os.getenv("GEMINI_API_KEY")
    try:
        key = key or st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if not key:
        st.error("API key missing. Set GEMINI_API_KEY (see steps).")
        st.stop()
    return genai.Client(api_key=key)


def parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def analyse(client, description, image, name, address, city, language):
    parts = []
    if image is not None:
        parts.append(types.Part.from_bytes(data=image.getvalue(), mime_type=image.type))
    parts.append(
        f"Complainant name: {name or '[Your Name]'}\n"
        f"Address / location of problem: {address or '[Location]'}\n"
        f"City: {city or '[City]'}\n"
        f"Today's date: {date.today().strftime('%d %B %Y')}\n"
        f"Letter language: {language}\n"
        f"Problem description: {description or '(see photo)'}"
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        max_output_tokens=4000,
    )
    last_error = None
    for model in MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=parts, config=config)
            return parse_json(resp.text)
        except Exception as e:  # try the next model
            last_error = e
    raise last_error


# ---------- UI ----------
st.set_page_config(page_title="Civic Complaint Router", page_icon="🏛️")
st.title("🏛️ Civic Complaint Router")
st.caption("Describe a local problem or upload a photo. Get the right department and a ready-to-send complaint letter.")

with st.sidebar:
    st.header("Your details")
    name = st.text_input("Your name")
    city = st.text_input("City / Town")
    address = st.text_input("Exact location of the problem")
    language = st.selectbox("Letter language", ["English", "Hindi", "Bengali"])

description = st.text_area(
    "What is the problem?",
    placeholder="Example: The streetlight near the bus stop has been off for 10 days and it is dangerous at night.",
)
photo = st.file_uploader("Photo of the problem (optional)", type=["jpg", "jpeg", "png"])
if photo:
    st.image(photo, width=300)

if st.button("Generate complaint", type="primary"):
    if not description and not photo:
        st.warning("Please describe the problem or upload a photo.")
    else:
        client = get_client()
        with st.spinner("Analysing your problem..."):
            try:
                result = analyse(client, description, photo, name, address, city, language)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        c1, c2 = st.columns(2)
        c1.metric("Category", result["category"])
        c2.metric("Urgency", result["urgency"])
        st.info(f"**Send to:** {result['department']}\n\n{result['reason']}")

        st.subheader("Complaint letter")
        st.text_input("Subject", result["subject_line"])
        letter = st.text_area("Letter (you can edit)", result["letter"], height=350)
        st.download_button("Download letter (.txt)", letter, file_name="complaint_letter.txt")

        st.subheader("What to do next")
        for s in result["next_steps"]:
            st.write(f"- {s}")

        st.caption(
            "Tip: Also check your city's official grievance app or website, and for central government "
            "matters the CPGRAMS portal (pgportal.gov.in). Verify department names for your area."
        )
