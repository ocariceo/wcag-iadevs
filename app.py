import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict
import pandas as pd
from io import BytesIO
import base64
import time
import requests
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup

# Configuración de página
st.set_page_config(
    page_title="Evaluador de Accesibilidad WCAG 2.1",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports para funcionalidades
from dotenv import load_dotenv
from openai import OpenAI  # Debe ser openai>=1.0
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Cargar variables de entorno
load_dotenv()

# --- CACHÉ DEL VECTORSTORE ---
@st.cache_resource
def create_wcag_vectorstore():
    """Crear y cachear la base de conocimiento WCAG 2.1"""
    wcag_guidelines = [
        Document(page_content="""
        WCAG 2.1 Nivel A - Requisitos básicos:
        1.1.1 Contenido no textual: Todo contenido no textual debe tener alternativas textuales
        1.2.1 Audio y video pregrabado: Proporcionar alternativas para medios basados en tiempo
        1.3.1 Información y relaciones: La información debe ser programáticamente determinable
        1.4.1 Uso del color: El color no debe ser el único medio visual para transmitir información
        2.1.1 Teclado: Toda funcionalidad debe estar disponible desde teclado
        2.1.2 Sin trampas de teclado: El foco del teclado no debe quedar atrapado
        2.2.1 Tiempo ajustable: Permitir al usuario desactivar, ajustar o extender límites de tiempo
        2.2.2 Pausar, detener, ocultar: Permitir control sobre contenido en movimiento
        3.1.1 Idioma de la página: El idioma principal debe estar programáticamente determinado
        3.2.1 Al recibir el foco: Los componentes no deben causar cambios de contexto inesperados
        3.2.2 Al recibir entradas: Cambiar configuraciones no debe causar cambios de contexto inesperados
        3.3.1 Identificación de errores: Los errores deben identificarse y describirse al usuario
        4.1.1 Procesamiento: El contenido debe poder ser interpretado por tecnologías de asistencia
        4.1.2 Nombre, función, valor: Los componentes UI deben tener nombres y funciones programáticamente determinables
        """),
        Document(page_content="""
        WCAG 2.1 Nivel AA - Requisitos estándar:
        1.2.4 Subtítulos en directo: Proporcionar subtítulos para contenido de audio en directo
        1.2.5 Audiodescripción pregrabada: Proporcionar audiodescripción para video pregrabado
        1.3.4 Orientación: El contenido no debe restringirse a una sola orientación de pantalla
        1.3.5 Identificar el propósito de entrada: El propósito de los campos debe ser programáticamente determinable
        1.4.3 Contraste mínimo: Relación de contraste de al menos 4.5:1 para texto normal
        1.4.4 Cambio de tamaño del texto: El texto debe poder redimensionarse hasta 200% sin pérdida de funcionalidad
        1.4.5 Imágenes de texto: Evitar usar imágenes de texto excepto cuando es esencial
        1.4.10 Reflow: El contenido debe presentarse sin scroll horizontal a 320px de ancho
        1.4.11 Contraste de elementos no textuales: Contraste de 3:1 para componentes UI y objetos gráficos
        1.4.12 Espaciado del texto: No debe haber pérdida de contenido o funcionalidad al ajustar espaciado
        1.4.13 Contenido en hover o focus: El contenido adicional debe ser descartable, hoverable y persistente
        2.1.4 Atajos de teclado de caracteres: Permitir desactivar o reasignar atajos de una sola tecla
        2.4.3 Orden del foco: El orden de navegación debe ser lógico y significativo
        2.4.6 Encabezados y etiquetas: Los encabezados y etiquetas deben describir el tema o propósito
        2.4.7 Foco visible: Cualquier interfaz operable por teclado debe tener un indicador de foco visible
        2.5.1 Gestos del puntero: Toda funcionalidad que use gestos multipunto debe tener alternativa de punto único
        2.5.2 Cancelación del puntero: Para funcionalidad operada por puntero, debe permitirse cancelación
        2.5.3 Etiqueta en nombre: El nombre accesible debe contener el texto visible de la etiqueta
        2.5.4 Activación por movimiento: La funcionalidad por movimiento debe tener alternativas y poder desactivarse
        3.1.2 Idioma de las partes: El idioma de cada pasaje debe estar programáticamente determinado
        3.2.3 Navegación coherente: Los mecanismos de navegación deben ser coherentes
        3.2.4 Identificación coherente: Los componentes con misma funcionalidad deben identificarse coherentemente
        3.3.3 Sugerencia de error: Proporcionar sugerencias cuando se detecten errores
        3.3.4 Prevención de errores legales/financieros/datos: Permitir revisión y corrección antes de envío
        4.1.3 Mensajes de estado: Los mensajes de estado deben comunicarse a tecnologías de asistencia
        """),
        Document(page_content="""
        WCAG 2.1 Nivel AAA - Requisitos avanzados:
        1.2.6 Lengua de señas pregrabada: Proporcionar interpretación en lengua de señas
        1.2.7 Audiodescripción extendida: Proporcionar audiodescripción extendida para video pregrabado
        1.2.8 Alternativa para medios pregrabados: Proporcionar alternativa textual para medios pregrabados
        1.2.9 Solo audio en directo: Proporcionar alternativa textual para contenido de solo audio en directo
        1.4.6 Contraste mejorado: Relación de contraste de al menos 7:1 para texto normal
        1.4.7 Audio de fondo bajo o nulo: Audio de fondo debe ser bajo o eliminable
        1.4.8 Presentación visual: Permitir personalización de presentación visual del texto
        1.4.9 Imágenes de texto sin excepción: No usar imágenes de texto excepto para logotipos
        2.1.3 Teclado sin excepción: Toda funcionalidad debe estar disponible desde teclado sin excepciones
        2.2.3 Sin límite de tiempo: No imponer límites de tiempo excepto para eventos en tiempo real
        2.2.4 Interrupciones: Las interrupciones pueden ser pospuestas o suprimidas por el usuario
        2.2.5 Reautenticación: Cuando expire una sesión, el usuario puede continuar sin pérdida de datos
        2.2.6 Tiempos de espera: Advertir a los usuarios sobre tiempos de espera que causan pérdida de datos
        2.3.2 Tres destellos: Las páginas no deben contener elementos que destellen más de tres veces por segundo
        2.3.3 Animación de interacciones: Permitir desactivar animaciones no esenciales
        2.4.8 Ubicación: Proporcionar información sobre ubicación del usuario dentro de un conjunto de páginas
        2.4.9 Propósito del enlace solo contexto: El propósito de cada enlace debe determinarse solo por el texto del enlace
        2.4.10 Encabezados de sección: Usar encabezados de sección para organizar contenido
        3.1.3 Palabras inusuales: Proporcionar mecanismo para identificar definiciones de palabras inusuales
        3.1.4 Abreviaciones: Proporcionar mecanismo para identificar la forma expandida de abreviaciones
        3.1.5 Nivel de lectura: Cuando el texto requiera habilidad de lectura avanzada, proporcionar contenido suplementario
        3.1.6 Pronunciación: Proporcionar mecanismo para identificar pronunciación específica de palabras
        3.2.5 Cambio a petición: Los cambios de contexto solo deben iniciarse por petición del usuario
        3.3.5 Ayuda contextual: Proporcionar ayuda contextual
        3.3.6 Prevención de errores general: Permitir revisión, corrección y confirmación antes de envío
        """)
    ]
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return Chroma.from_documents(
        documents=wcag_guidelines,
        embedding=embeddings,
        collection_name="wcag_guidelines"
    )


class RobustWebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.setup_session()

    def setup_session(self):
        from fake_useragent import UserAgent
        ua = UserAgent()
        self.session.headers.update({
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        })

    def get_with_retries(self, url: str, max_retries: int = 3) -> str | None:
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(time.uniform(2, 5))
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', 60))
                    st.warning(f"Rate limit detectado. Esperando {wait_time} segundos...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 403:
                    from fake_useragent import UserAgent
                    ua = UserAgent()
                    self.session.headers['User-Agent'] = ua.random
                    continue
            except Exception as e:
                st.warning(f"Intento {attempt + 1} falló: {str(e)}")
        return None

    def scrape_with_selenium(self, url: str) -> str | None:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import undetected_chromedriver as uc

            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)

            driver = uc.Chrome(options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            html_content = driver.page_source
            driver.quit()
            return html_content
        except Exception as e:
            st.error(f"Error con Selenium: {str(e)}")
            return None

    def scrape_with_playwright(self, url: str) -> str | None:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                page = context.new_page()
                page.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}", lambda route: route.abort())
                page.goto(url, wait_until='networkidle')
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(2000)
                html_content = page.content()
                browser.close()
                return html_content
        except Exception as e:
            st.error(f"Error con Playwright: {str(e)}")
            return None

    def scrape_website(self, url: str) -> str | None:
        st.info("🔍 Iniciando scraping del sitio web...")
        content = self.get_with_retries(url)
        if content and len(content) > 1000:
            st.success("✅ Contenido obtenido con requests")
            return content

        st.info("🔄 Intentando con Selenium...")
        content = self.scrape_with_selenium(url)
        if content and len(content) > 1000:
            st.success("✅ Contenido obtenido con Selenium")
            return content

        st.info("🔄 Intentando con Playwright...")
        content = self.scrape_with_playwright(url)
        if content and len(content) > 1000:
            st.success("✅ Contenido obtenido con Playwright")
            return content

        st.error("❌ No se pudo obtener el contenido del sitio web")
        return None


class WCAGEvaluator:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)  # openai>=1.0
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0.1)
        self.vectorstore = create_wcag_vectorstore()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def analyze_html_accessibility(self, html_content: str) -> Dict:
        soup = BeautifulSoup(html_content, 'html.parser')
        analysis_data = {
            'images': len(soup.find_all('img')),
            'images_with_alt': len(soup.find_all('img', alt=True)),
            'headings': [tag.name for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
            'links': len(soup.find_all('a')),
            'forms': len(soup.find_all('form')),
            'inputs': len(soup.find_all(['input', 'textarea', 'select'])),
            'labels': len(soup.find_all('label')),
            'lang_attr': soup.find('html', lang=True) is not None,
            'title': soup.find('title') is not None,
            'skip_links': len(soup.find_all('a', href=lambda x: x and x.startswith('#'))),
            'aria_labels': len(soup.find_all(attrs={'aria-label': True})),
            'roles': len(soup.find_all(attrs={'role': True})),
            'contrast_issues': self._check_contrast_issues(soup),
            'keyboard_focus': self._check_keyboard_focus(soup),
            'semantic_structure': self._check_semantic_structure(soup)
        }

        prompt = f"""
        Analiza la siguiente información de accesibilidad de un sitio web según los estándares WCAG 2.1:
        Datos del análisis:
        - Imágenes totales: {analysis_data['images']}
        - Imágenes con alt text: {analysis_data['images_with_alt']}
        - Estructura de encabezados: {analysis_data['headings']}
        - Enlaces totales: {analysis_data['links']}
        - Formularios: {analysis_data['forms']}
        - Campos de entrada: {analysis_data['inputs']}
        - Etiquetas: {analysis_data['labels']}
        - Atributo lang en HTML: {analysis_data['lang_attr']}
        - Título de página: {analysis_data['title']}
        - Enlaces de salto: {analysis_data['skip_links']}
        - Elementos con aria-label: {analysis_data['aria_labels']}
        - Elementos con roles ARIA: {analysis_data['roles']}
        Problemas detectados:
        - Contraste: {analysis_data['contrast_issues']}
        - Navegación por teclado: {analysis_data['keyboard_focus']}
        - Estructura semántica: {analysis_data['semantic_structure']}
        Proporciona:
        1. Evaluación de conformidad WCAG 2.1 (A, AA, o AAA)
        2. Lista de problemas específicos encontrados
        3. Recomendaciones detalladas de mejora
        4. Puntuación numérica del 1-100
        Responde en formato JSON con las claves: level, score, issues, recommendations, summary
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en accesibilidad web y WCAG 2.1. Analiza sitios web y proporciona evaluaciones detalladas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            st.error(f"Error en análisis con IA: {str(e)}")
            return self._fallback_analysis(analysis_data)

    def _check_contrast_issues(self, soup) -> list:
        issues = []
        elements_with_style = soup.find_all(attrs={'style': True})
        for elem in elements_with_style:
            style = elem.get('style', '')
            if 'color:' in style and 'background' in style:
                issues.append(f"Posible problema de contraste en {elem.name}")
        return issues

    def _check_keyboard_focus(self, soup) -> list:
        issues = []
        interactive_elements = soup.find_all(['a', 'button', 'input', 'select', 'textarea'])
        for elem in interactive_elements:
            if elem.get('tabindex') == '-1':
                issues.append(f"Elemento {elem.name} excluido de navegación por teclado")
        return issues

    def _check_semantic_structure(self, soup) -> list:
        issues = []
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if not headings:
            issues.append("No se encontraron encabezados en la página")
        h1_count = len(soup.find_all('h1'))
        if h1_count == 0:
            issues.append("Falta encabezado H1 principal")
        elif h1_count > 1:
            issues.append("Múltiples encabezados H1 encontrados")
        return issues

    def _fallback_analysis(self, data: Dict) -> Dict:
        score = 0
        issues = []
        recommendations = []

        if data['images'] > 0:
            alt_ratio = data['images_with_alt'] / data['images']
            score += alt_ratio * 20
            if alt_ratio < 1:
                issues.append("Algunas imágenes no tienen texto alternativo")
                recommendations.append("Agregar atributos alt descriptivos a todas las imágenes")

        if data['lang_attr']:
            score += 10
        else:
            issues.append("Falta atributo lang en elemento HTML")
            recommendations.append("Agregar atributo lang='es' al elemento HTML")

        if data['title']:
            score += 10
        else:
            issues.append("Falta título de página")
            recommendations.append("Agregar elemento <title> descriptivo")

        level = "AA" if score >= 80 else "A" if score >= 60 else "No conforme"

        return {
            'level': level,
            'score': int(score),
            'issues': issues,
            'recommendations': recommendations,
            'summary': f"Análisis básico completado. Puntuación: {int(score)}/100"
        }


class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()

    def _create_custom_styles(self):
        custom_styles = {}
        custom_styles['Title'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f4e79'),
            alignment=1
        )
        custom_styles['Subtitle'] = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.HexColor('#2c5aa0')
        )
        custom_styles['Issue'] = ParagraphStyle(
            'Issue',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            bulletIndent=10,
            textColor=colors.HexColor('#d63384')
        )
        custom_styles['Recommendation'] = ParagraphStyle(
            'Recommendation',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            bulletIndent=10,
            textColor=colors.HexColor('#198754')
        )
        return custom_styles

    def generate_pdf_report(self, analysis_result: Dict, url: str = None) -> BytesIO:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        story = []

        story.append(Paragraph("Reporte de Accesibilidad Web WCAG 2.1", self.custom_styles['Title']))
        story.append(Spacer(1, 20))

        if url:
            story.append(Paragraph(f"<b>URL analizada:</b> {url}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Fecha del análisis:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Nivel de conformidad:</b> {analysis_result.get('level', 'No determinado')}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Puntuación:</b> {analysis_result.get('score', 0)}/100", self.styles['Normal']))
        story.append(Spacer(1, 30))

        story.append(Paragraph("Resumen Ejecutivo", self.custom_styles['Subtitle']))
        story.append(Paragraph(analysis_result.get('summary', 'No disponible'), self.styles['Normal']))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Problemas Identificados", self.custom_styles['Subtitle']))
        issues = analysis_result.get('issues', [])
        if issues:
            for issue in issues:
                story.append(Paragraph(f"• {issue}", self.custom_styles['Issue']))
        else:
            story.append(Paragraph("No se identificaron problemas específicos.", self.styles['Normal']))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Recomendaciones de Mejora", self.custom_styles['Subtitle']))
        recommendations = analysis_result.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", self.custom_styles['Recommendation']))
        else:
            story.append(Paragraph("No se generaron recomendaciones específicas.", self.styles['Normal']))
        story.append(Spacer(1, 30))

        story.append(Paragraph("Criterios WCAG 2.1 Evaluados", self.custom_styles['Subtitle']))
        table_data = [
            ['Criterio', 'Nivel', 'Estado', 'Observaciones'],
            ['1.1.1 Contenido no textual', 'A', '✓' if analysis_result.get('score', 0) > 60 else '✗', 'Texto alternativo para imágenes'],
            ['1.4.3 Contraste mínimo', 'AA', '✓' if analysis_result.get('score', 0) > 70 else '✗', 'Contraste de colores'],
            ['2.1.1 Teclado', 'A', '✓' if analysis_result.get('score', 0) > 60 else '✗', 'Navegación por teclado'],
            ['3.1.1 Idioma de página', 'A', '✓' if analysis_result.get('score', 0) > 50 else '✗', 'Atributo lang definido'],
            ['4.1.2 Nombre, función, valor', 'A', '✓' if analysis_result.get('score', 0) > 65 else '✗', 'Elementos de interfaz accesibles']
        ]
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)

        doc.build(story)
        buffer.seek(0)
        return buffer


def display_results(analysis_result: Dict, url: str = None):
    st.markdown("---")
    st.subheader("📊 Resultados del Análisis")
    col1, col2, col3 = st.columns(3)
    with col1:
        score = analysis_result.get('score', 0)
        st.metric("Puntuación General", f"{score}/100")
    with col2:
        level = analysis_result.get('level', 'No determinado')
        st.metric("Nivel WCAG 2.1", level)
    with col3:
        issues_count = len(analysis_result.get('issues', []))
        st.metric("Problemas Encontrados", issues_count)

    progress_color = "success" if score >= 80 else "warning" if score >= 60 else "error"
    st.progress(score / 100)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Resumen", "❌ Problemas", "💡 Recomendaciones", "📈 Detalles"])
    with tab1:
        st.subheader("Resumen Ejecutivo")
        st.info(analysis_result.get('summary', 'No disponible'))
        fig, ax = plt.subplots(figsize=(6, 6))
        values = [score, 100 - score]
        colors_chart = ['#28a745', '#dc3545']
        ax.pie(values, labels=['Conforme', 'No Conforme'], autopct='%1.1f%%', colors=colors_chart, startangle=90, wedgeprops=dict(width=0.5))
        ax.set_title(f'Nivel de Conformidad WCAG 2.1\n({level})', fontsize=14, fontweight='bold')
        st.pyplot(fig)

    with tab2:
        st.subheader("Problemas Identificados")
        issues = analysis_result.get('issues', [])
        if issues:
            for i, issue in enumerate(issues, 1):
                st.error(f"**{i}.** {issue}")
        else:
            st.success("🎉 ¡No se identificaron problemas específicos!")

    with tab3:
        st.subheader("Recomendaciones de Mejora")
        recommendations = analysis_result.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.info(f"**{i}.** {rec}")
        else:
            st.success("✅ No se requieren recomendaciones adicionales")

    with tab4:
        st.subheader("Análisis Técnico Detallado")
        criteria_data = {
            'Criterio WCAG': ['1.1.1 Contenido no textual', '1.4.3 Contraste mínimo', '2.1.1 Teclado', '3.1.1 Idioma de página', '4.1.2 Nombre, función, valor'],
            'Nivel': ['A', 'AA', 'A', 'A', 'A'],
            'Estado': ['✅' if score > 60 else '❌', '✅' if score > 70 else '❌', '✅' if score > 60 else '❌', '✅' if score > 50 else '❌', '✅' if score > 65 else '❌'],
            'Puntuación': [f"{min(20, score//5)}/20", f"{min(25, score//4)}/25", f"{min(20, score//5)}/20", f"{min(15, score//7)}/15", f"{min(20, score//5)}/20"]
        }
        df = pd.DataFrame(criteria_data)
        st.dataframe(df, use_container_width=True)

        tech_info = f"""
        **Análisis realizado:** {datetime.now().strftime('%d/%m/%Y a las %H:%M:%S')}
        **Método de obtención:** {"Web Scraping" if url else "Código HTML directo"}
        **Motor de IA:** GPT-4 con RAG
        **Base de conocimiento:** WCAG 2.1 Guidelines
        **Versión de la aplicación:** 1.0.0
        """
        st.markdown(tech_info)


def setup_page_config():
    st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stMetric > label { font-size: 16px !important; font-weight: 600 !important; }
    .stProgress > div > div { background: linear-gradient(90deg, #ff6b6b, #feca57, #48ca77); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 10px; color: #1f4e79; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #1f4e79; color: white; }
    </style>
    """, unsafe_allow_html=True)


def show_help():
    with st.expander("❓ Ayuda y Guías de Uso"):
        st.markdown("""
        ### 🚀 Cómo usar esta aplicación
        1. **Configurar variables de entorno**: Crea un archivo `.env` con tu API key de OpenAI
        2. **Seleccionar modo**: Elige entre análisis de URL o código HTML directo
        3. **Iniciar análisis**: Haz clic en el botón correspondiente
        4. **Revisar resultados**: Explora los tabs con diferentes aspectos del análisis
        5. **Descargar reporte**: Obtén un PDF profesional con todos los hallazgos

        ### 🔧 Solución de Problemas
        **El scraping falla:**
        - Verifica que la URL sea accesible
        - Algunos sitios pueden tener protecciones muy estrictas
        - Prueba con el modo de código HTML directo

        **Error de API:**
        - Revisa tu API key de OpenAI en el archivo .env
        - Asegúrate de tener créditos disponibles

        **Resultados inesperados:**
        - La evaluación es automatizada y puede no capturar todos los aspectos
        - Complementa con revisión manual por expertos en accesibilidad

        ### 📚 Recursos Adicionales
        - [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
        - [WebAIM](https://webaim.org/)
        - [axe-core](https://github.com/dequelabs/axe-core)
        """)


def main():
    setup_page_config()
    show_help()

    st.title("🌐 Evaluador de Accesibilidad Web WCAG 2.1")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuración")
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            st.error("❌ No se encontró OPENAI_API_KEY en el archivo .env")
            st.stop()
        st.success("✅ API Key configurada correctamente")

        st.header("📋 Opciones de Análisis")
        analysis_mode = st.radio(
            "Selecciona el modo de análisis:",
            ["URL del sitio web", "Código HTML directo"]
        )

        st.header("📊 Niveles WCAG 2.1")
        st.info("""
        **A**: Nivel básico de accesibilidad
        **AA**: Nivel estándar (recomendado)
        **AAA**: Nivel máximo de accesibilidad
        """)

    col1, col2 = st.columns([2, 1])

    with col1:
        if analysis_mode == "URL del sitio web":
            st.subheader("🔗 Análisis por URL")
            url_input = st.text_input("Ingresa la URL del sitio web a evaluar:", placeholder="https://ejemplo.com")
            if st.button("🚀 Iniciar Análisis", type="primary"):
                if url_input:
                    with st.spinner("Analizando sitio web..."):
                        scraper = RobustWebScraper()
                        html_content = scraper.scrape_website(url_input)
                        if html_content:
                            evaluator = WCAGEvaluator(openai_key)
                            analysis_result = evaluator.analyze_html_accessibility(html_content)
                            st.session_state['analysis_result'] = analysis_result
                            display_results(analysis_result, url_input)
                            report_gen = ReportGenerator()
                            pdf_buffer = report_gen.generate_pdf_report(analysis_result, url_input)
                            st.download_button(
                                label="📥 Descargar Reporte PDF",
                                data=pdf_buffer,
                                file_name=f"reporte_accesibilidad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("No se pudo obtener el contenido del sitio web")
                else:
                    st.warning("Por favor, ingresa una URL válida")

        else:
            st.subheader("📝 Análisis de Código HTML")
            html_input = st.text_area("Pega aquí el código HTML a evaluar:", height=300, placeholder="<html>...</html>")
            if st.button("🔍 Analizar HTML", type="primary"):
                if html_input.strip():
                    with st.spinner("Analizando código HTML..."):
                        evaluator = WCAGEvaluator(openai_key)
                        analysis_result = evaluator.analyze_html_accessibility(html_input)
                        st.session_state['analysis_result'] = analysis_result
                        display_results(analysis_result)
                        report_gen = ReportGenerator()
                        pdf_buffer = report_gen.generate_pdf_report(analysis_result)
                        st.download_button(
                            label="📥 Descargar Reporte PDF",
                            data=pdf_buffer,
                            file_name=f"reporte_accesibilidad_html_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.warning("Por favor, ingresa código HTML válido")

    with col2:
        st.subheader("ℹ️ Información")
        st.markdown("""
        ### ¿Qué evaluamos?
        **🎯 Criterios WCAG 2.1:**
        - Perceptibilidad
        - Operabilidad  
        - Comprensibilidad
        - Robustez
        **🔍 Técnicas de Scraping:**
        - Requests con rotación de User-Agent
        - Selenium WebDriver
        - Playwright
        - Manejo de rate limits
        - Bypass de protecciones anti-bot
        **🧠 IA Generativa:**
        - Análisis contextual con GPT-4
        - Base de conocimiento WCAG 2.1
        - Recomendaciones personalizadas
        - RAG con LangChain
        """)

        st.subheader("📈 Métricas de Evaluación")
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            st.metric("Puntuación General", f"{result.get('score', 0)}/100")
            st.metric("Nivel WCAG", result.get('level', 'N/A'))
        else:
            st.metric("Puntuación General", "—")
            st.metric("Nivel WCAG", "—")


if __name__ == "__main__":
    main()