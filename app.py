from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import os, smtplib, ssl
from email.message import EmailMessage
from email.utils import formataddr
from secrets import token_urlsafe

# --------------------------
# Config
# --------------------------
FEATURES = [
    "MONTANT_PRIME_TOTALE_VALUE",
    "CODE_SEXE",
    "STATUT_MATRIMONIAL",
    "NATURE_DOSSIER_CONTRAT",
    "PRIME_ANORMALE",
    "ANCIENNETE",
    "PACK",
]

SEUIL_RISQUE = 0.50   # 50%
MARGE = 0.30          # pour CLV simplifiée

# Config e-mail
MAIL_SERVER   = os.getenv("MAIL_SERVER",   "smtp.gmail.com")
MAIL_PORT     = int(os.getenv("MAIL_PORT", "587"))
MAIL_USERNAME = os.getenv("MAIL_USERNAME", "amirabelhay34@gmail.com")
MAIL_PASSWORD = (os.getenv("MAIL_PASSWORD", "arav xlgk wzit frop") or "").replace(" ", "")  # <- sans espaces
MAIL_TO       = os.getenv("MAIL_TO",       "profeeder12345678@gmail.com")
MAIL_USE_TLS  = True


def send_mail(subject: str,
              html_body: str,
              text_body: str = "",
              to_address: str = None,
              reply_to: str = None,
              from_name: str = None):
    """Envoie un e-mail via SMTP (TLS)."""
    msg = EmailMessage()
    # From affiché (le compte réel utilisé reste MAIL_USERNAME)
    msg["From"] = formataddr((from_name or "COMAR", MAIL_USERNAME))
    msg["To"] = to_address or MAIL_TO
    msg["Subject"] = subject
    if reply_to:
        msg["Reply-To"] = reply_to
    if text_body:
        msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
        if MAIL_USE_TLS:
            server.starttls(context=context)
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.send_message(msg)

# --------------------------
# Helpers
# --------------------------
def to_float_safe(x: str, default=0.0):
    if x is None:
        return default
    x = x.strip()
    if not x:
        return default
    try:
        return float(x.replace(",", "."))
    except Exception:
        return default

def align_columns_to_model(df: pd.DataFrame, expected_cols):
    """Réordonne et ajoute les colonnes manquantes pour coller au pipeline."""
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    return df[expected_cols]

# --------------------------
# App & modèle
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates", static_folder="static")

# SECRET_KEY pour sessions/flash
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", token_urlsafe(32))

MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
try:
    pipe = joblib.load(MODEL_PATH)
    print("[OK] Modèle chargé :", type(pipe))
except Exception as e:
    pipe = None
    print(f"[ERREUR] Impossible de charger le modèle : {e}")

# --------------------------
# Routes
# --------------------------

# Accueil
@app.route("/")
def home():
    return render_template("home.html")

# Contact
@app.route("/contact")
def contact():
    sent = request.args.get("sent", type=int)  # récupère ?sent=1 après envoi
    infos = {
        "adresse": "COMAR Avenue Habib Bourguiba 1001 Tunis R.P",
        "tel": "+216 71 340 899",
        "service_client": "82 100 001",
        "fax": "+216 71 344 778",
        "email": "contact@comar.tn",
        "maps_url": "https://www.google.com/maps/search/?api=1&query=COMAR+Avenue+Habib+Bourguiba+1001+Tunis"
    }
    return render_template("contact.html", infos=infos, sent=sent)

# Proposition
@app.route("/proposition", methods=["GET", "POST"])
def proposition():
    """
    Formulaire employé => propose une action (remise, appel, upgrade pack...).
    - Si on arrive depuis la prédiction (query string): pré-rempli
    - Sinon (arrivée directe): vide
    """
    if request.method == "POST":
        employe_nom   = request.form.get("employe_nom", "").strip()
        employe_mail  = request.form.get("employe_mail", "").strip()
        client_id     = request.form.get("client_id", "").strip()
        proba         = request.form.get("proba", "").strip()
        classe        = request.form.get("classe", "").strip()
        pack          = request.form.get("pack", "").strip()
        prime         = request.form.get("prime", "").strip()
        anciennete    = request.form.get("anciennete", "").strip()

        action        = request.form.get("action", "")
        remise        = request.form.get("remise", "")
        echeance      = request.form.get("echeance", "")
        commentaire   = request.form.get("commentaire", "").strip()

        errors = []
        if not employe_nom:  errors.append("Nom de l’employé obligatoire.")
        if not employe_mail: errors.append("E-mail de l’employé obligatoire.")
        if not client_id:    errors.append("Identifiant client obligatoire.")
        if not proba:        errors.append("Probabilité manquante.")

        if errors:
            return render_template("proposition.html", errors=errors, initial=request.form)

        html = render_template(
            "mail_proposition.html",
            employe_nom=employe_nom, employe_mail=employe_mail,
            client_id=client_id, proba=proba, classe=classe,
            pack=pack, prime=prime, anciennete=anciennete,
            action=action, remise=remise, echeance=echeance,
            commentaire=commentaire
        )

        subject = f"[COMAR • Proposition] Client {client_id} – risque {proba}% ({classe})"

        try:
            send_mail(
                subject,
                html_body=html,
                text_body=f"Proposition pour client {client_id}",
                to_address="amirabelhay34@gmail.com",  # destinataire final
                reply_to=employe_mail,                 # la réponse ira à l’employé
                from_name=employe_nom                  # nom affiché dans "From"
            )
            flash("Proposition envoyée à COMAR avec succès ✅", "success")
            return redirect(url_for("contact", sent=1))
        except Exception as e:
            flash(f"Échec de l’envoi de l’e-mail : {e}", "error")
            return render_template("proposition.html", errors=["Erreur d’envoi e-mail."], initial=request.form)

    # GET : préremplissage éventuel via query string
    initial = {
        "proba": request.args.get("proba", ""),
        "classe": request.args.get("classe", ""),
        "pack": request.args.get("pack", ""),
        "prime": request.args.get("prime", ""),
        "anciennete": request.args.get("anciennete", "")
    }
    return render_template("proposition.html", initial=initial)

# Prédiction
@app.route("/prediction", methods=["GET", "POST"])
def index():
    proba = None
    classe = None
    clv_estimee = None
    erreurs = []

    # (Option) Réafficher un résultat en GET si retour depuis /proposition
    if request.method == "GET" and request.args.get("back") == "1":
        try:
            proba_qs   = request.args.get("proba")
            classe_qs  = request.args.get("classe")
            montant_qs = request.args.get("montant")
            if proba_qs and classe_qs:
                proba = round(float(proba_qs), 2)
                p = max(min(proba / 100.0, 0.999999), 1e-6)
                montant_f = to_float_safe(montant_qs, default=0.0)
                clv_estimee = round(montant_f * MARGE * (1 / p), 2)
                classe = classe_qs
        except Exception as e:
            erreurs.append(str(e))

    if request.method == "POST":
        if pipe is None:
            erreurs.append("Modèle introuvable/illisible. Vérifie models/random_forest_model.pkl.")
        else:
            try:
                montant      = request.form.get("montant")
                code_sexe    = request.form.get("code_sexe")    or "Inconnu"
                statut_mat   = request.form.get("statut_mat")   or "Inconnu"
                nature       = request.form.get("nature")       or "Inconnu"
                prime_anorm  = request.form.get("prime_anorm")  or "Non"
                anciennete   = request.form.get("anciennete")
                pack         = request.form.get("pack")         or "RC"

                montant_f    = to_float_safe(montant, default=0.0)
                anciennete_f = to_float_safe(anciennete, default=0.0)

                data = pd.DataFrame([{
                    "MONTANT_PRIME_TOTALE_VALUE": montant_f,
                    "CODE_SEXE": str(code_sexe),
                    "STATUT_MATRIMONIAL": str(statut_mat),
                    "NATURE_DOSSIER_CONTRAT": str(nature),
                    "PRIME_ANORMALE": str(prime_anorm),
                    "ANCIENNETE": anciennete_f,
                    "PACK": str(pack),
                }])

                data = align_columns_to_model(data, FEATURES)
                data["MONTANT_PRIME_TOTALE_VALUE"] = data["MONTANT_PRIME_TOTALE_VALUE"].astype(float)
                data["ANCIENNETE"] = data["ANCIENNETE"].astype(float)
                for c in ["CODE_SEXE","STATUT_MATRIMONIAL","NATURE_DOSSIER_CONTRAT","PRIME_ANORMALE","PACK"]:
                    data[c] = data[c].astype(str)

                if hasattr(pipe, "predict_proba"):
                    p = float(pipe.predict_proba(data)[:, 1][0])
                else:
                    yhat = int(pipe.predict(data)[0])
                    p = 0.51 if yhat == 1 else 0.49

                proba = round(p * 100, 2)
                classe = "À risque" if p >= SEUIL_RISQUE else "Stable"
                clv_estimee = round(montant_f * MARGE * (1 / max(p, 1e-6)), 2)

            except Exception as e:
                erreurs.append(str(e))

    return render_template(
        "index.html",
        proba=proba,
        classe=classe,
        clv=clv_estimee,
        seuil=int(SEUIL_RISQUE * 100),
        erreurs=erreurs,
    )

# Détail produits Particuliers
@app.route("/produits/<slug>")
def produit_detail(slug):
    PRODUITS = {
        "auto": {
            "title": "Assurance Auto",
            "subtitle": "Roulez serein : RC obligatoire, protection du conducteur, assistance 24/7.",
            "hero": "img/produits/auto_hero.jpg",
            "bullets": [
                "Responsabilité Civile (obligatoire)",
                "Défense & recours",
                "Bris de glace, vol, incendie (selon formules)",
                "Dommages Collision / Tous Risques",
                "Assistance et remorquage 24/7",
                "Protection du conducteur",
            ],
            "cta": {"label": "Demander un devis", "href": url_for("contact")},
            "icon": "img/particuliers/auto.png",
        },
        "maison": {
            "title": "Assurance Habitation (MRH)",
            "subtitle": "Protégez votre logement et vos biens contre les sinistres du quotidien.",
            "hero": "img/produits/maison_hero.jpg",
            "bullets": [
                "Incendie, explosion",
                "Dégâts des eaux",
                "Vol & vandalisme",
                "Bris de glace",
                "Responsabilité Civile familiale",
                "Assistance habitation",
            ],
            "cta": {"label": "Demander un devis", "href": url_for("contact")},
            "icon": "img/particuliers/maison.png",
        },
        "famille": {
            "title": "Protection Famille",
            "subtitle": "Des garanties pour vos proches : scolaire, accidents de la vie, voyage…",
            "hero": "img/produits/famille_hero.jpg",
            "bullets": [
                "Assurance scolaire & extrascolaire",
                "Garantie Accidents de la Vie (GAV)",
                "Responsabilité Civile familiale",
                "Assurance voyage",
                "Services d’assistance",
            ],
            "cta": {"label": "Contacter un conseiller", "href": url_for("contact")},
            "icon": "img/particuliers/famille.png",
        },
        "sante": {
            "title": "Complémentaire Santé",
            "subtitle": "Remboursements renforcés : hospitalisation, soins courants, optique, dentaire.",
            "hero": "img/produits/sante_hero.jpg",
            "bullets": [
                "Hospitalisation",
                "Soins courants (consultations & pharmacie)",
                "Optique & dentaire",
                "Maternité (selon formules)",
                "Réseau de partenaires",
            ],
            "cta": {"label": "Demander un devis", "href": url_for("contact")},
            "icon": "img/particuliers/sante.png",
        },
    }

    p = PRODUITS.get(slug)
    if not p:
        return redirect(url_for("home"))
    return render_template("produit_detail.html", p=p, slug=slug)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
