from scrapy.item import Item, Field
from scrapy.loader.processors import MapCompose, TakeFirst, Join, Identity
from datetime import datetime
import re
import numpy as np

def do_strip(text):
    if text != None:
        text = ' '.join(text.strip().split())
    else:
        text = ''
    return text

def com_binary(text):
    text = text.lower()
    if bool(re.search('com ', text)):
        out = 1
    else:
        out = 0
    return out
    
def fuel_converter(text):
    text = text.lower()    
    if text == 'gasolina':
        text = 'gasoline'
    elif text == 'eléctrico':
        text = 'electric'
    elif text == 'híbrido (gasolina)':
        text = 'hybrid_gasoline'
    elif text == 'híbrido (diesel)':
        text = 'hybrid_diesel'
    return text

def month_converter(text):
    text = text.lower()
    text = text.replace('janeiro', 'january')
    text = text.replace('fevereiro', 'february')
    text = text.replace('março', 'march')
    text = text.replace('abril', 'april')
    text = text.replace('maio', 'may')
    text = text.replace('junho', 'june')
    text = text.replace('julho', 'july')
    text = text.replace('agosto', 'august')
    text = text.replace('setembro', 'september')
    text = text.replace('outubro', 'october')
    text = text.replace('novembro', 'november')
    text = text.replace('dezembro', 'december')
    return text

def numeric_parser(text):
    text = text.replace(' ','').replace(',','.')
    value = re.findall(r'\d+(?:\.\d+)?', text)
    if len(value) > 0:
        value = float(value[0])
    else:
        value = 0.0
    return value

def negociable_parser(text):
    text = text.lower()
    if bool(re.search('fixo',text)):
        value = 0
    else:
        value = 1
    return value

def convert_date(text):
    try:
        date = datetime.strptime(text, '%H:%M, %d %B %Y')
    except:
        date = ''
    return date

def convert_date2(text):
    try:
        date = datetime.strptime(text, '%d/%m/Y')
    except:
        date = ''
    return date

def region_parser(text):
    text = text.replace('(','').replace(')','')
    return text

def origin_parser(text):
    text = text.split(' ')
    if len(text) > 1:
        text = text[1].lower()
    else:
        text = ''
    if text == 'nacional':
        text = 'national'
    elif text == 'importado':
        text = 'imported'
    return text

def advertiser_parser(text):
    if text == 'Particular':
        text = 'individial'
    elif text == 'Profissional':
        text = 'professional'
    else:
        text = ''
    return text

def brand_parser(text):
    if text == 'Outra não listada':
        text = 'other'
    return text

def segment_parser(text):
    text = text.replace('Coupé','coupe')
    text = text.replace('Utilitário','utilitary')
    text = text.replace('Carrinha','van')
    text = text.replace('Monovolume','mini_van')
    text = text.replace('Pequeno citadino','city_small')
    text = text.replace('Citadino','city')
    text = text.replace('SUV / TT','suv')
    text = text.replace('Sedan','sedan')
    text = text.replace('Cabrio','cabrio')
    return text

def color_parser(text):
    text = text.replace('Branco','white')
    text = text.replace('Cinzento','gray')
    text = text.replace('Azul','blue')
    text = text.replace('Preto','black')
    text = text.replace('Prateado','silver')
    text = text.replace('Castanho','brown')
    text = text.replace('Vermelho','red')
    text = text.replace('Laranja','orange')
    text = text.replace('Verde','green')
    text = text.replace('Outra','other')
    text = text.replace('Bege','beige')
    text = text.replace('Roxo','purple')
    text = text.replace('Dourado','golden')
    text = text.replace('Amarelo','yellow')
    return text

def sim_binary(text):
    if bool(re.search('sim', text.lower())):
        out = 1
    else:
        out = 0
    return out

def geartype_parser(text):
    text = text.lower()
    if text == 'automática':
        text = 'automatic'
    elif text == 'Semi-automática':
        text = 'semiauto'
    return text

def traction_parser(text):
    if text == 'Tracção traseira':
        text = 'back'
    elif text == 'Tracção dianteira':
        text = 'front'
    elif text == 'Integral':
        text = 'both'
    return text

def consumption_parser(text):
    text = text.split(' ')
    if len(text) > 1:
        value = numeric_parser(text[0])
    else:
        value = np.nan
    return value

def ceiling_parser(text):
    if text == 'Tecto de Abrir Panorâmico':
        text = 'panoramic'
    elif text == 'Tecto de Abrir Elétrico':
        text = 'electric'
    elif text == 'Tecto de Abrir Manual':
        text = 'manual'
    return text


def upholstery_parser(text):
    if text == 'Estofos de Tecido':
        text = 'fabric'
    elif text == 'Estofos de Pele':
        text = 'leather'
    return text

def aircon_parser(text):
    if text == 'AC Automático':
        text = 'automatic'
    elif text == 'AC Manual':
        text = 'manual'
    elif text == 'AC Independente':
        text = 'independent'
    return text

def used_parser(text):
    if text == 'Usados':
        text = 'used'
    elif text == 'Novos':
        text = 'new'
    return text

def canopy_parser(text):
    if text == 'Capota de Lona':
        text = 'canvas'
    elif text == 'Capota Rígida':
        text = 'rigid'
    elif text == 'Capota Hardtop':
        text = 'hardtop'
    return text


class CarItem(Item):
    guaranty = Field(
        input_processor=MapCompose(do_strip, com_binary),
        output_processor=TakeFirst()
    )

    title = Field(
        input_processor=MapCompose(do_strip),
        output_processor=TakeFirst()
    )

    fuel = Field(
        input_processor=MapCompose(do_strip, fuel_converter),
        output_processor=TakeFirst()
    )

    first_registration_month = Field(
        input_processor=MapCompose(do_strip, month_converter),
        output_processor=TakeFirst()
    )

    first_registration_year = Field(
        input_processor=MapCompose(do_strip, int),
        output_processor=TakeFirst()
    )

    mileage = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    power = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    price = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    negociable = Field(
        input_processor=MapCompose(do_strip, negociable_parser),
        output_processor=TakeFirst()
    )

    city = Field(
        input_processor=MapCompose(do_strip),
        output_processor=TakeFirst()
    )

    region = Field(
        input_processor=MapCompose(do_strip, region_parser),
        output_processor=TakeFirst()
    )

    origin = Field(
        input_processor=MapCompose(do_strip, origin_parser),
        output_processor=TakeFirst()
    )

    price_detail = Field(
        input_processor=MapCompose(do_strip, com_binary),
        output_processor=TakeFirst()
    )

    post_date = Field(
        input_processor=MapCompose(do_strip, month_converter, convert_date),
        output_processor=TakeFirst()
    )

    link = Field(
        input_processor=MapCompose(do_strip),
        output_processor=TakeFirst()
    )

    advertiser = Field(
        input_processor=MapCompose(do_strip, advertiser_parser),
        output_processor=TakeFirst()
    )

    brand = Field(
        input_processor=MapCompose(do_strip, brand_parser),
        output_processor=TakeFirst()
    )

    model = Field(
        input_processor=MapCompose(do_strip),
        output_processor=TakeFirst()
    )

    series = Field(
        input_processor=MapCompose(do_strip),
        output_processor=TakeFirst()
    )
    
    version = Field(
        input_processor=MapCompose(do_strip),
        output_processor=TakeFirst()
    )

    cylinder = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    segment = Field(
        input_processor=MapCompose(do_strip, segment_parser),
        output_processor=TakeFirst()
    )

    color = Field(
        input_processor=MapCompose(do_strip, color_parser),
        output_processor=TakeFirst()
    )

    metallic = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    gear_type = Field(
        input_processor=MapCompose(do_strip, geartype_parser),
        output_processor=TakeFirst()
    )

    gears_n = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    doors_n = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )
    
    capacity = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    car_class = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )
    
    traction = Field(
        input_processor=MapCompose(do_strip, traction_parser),
        output_processor=TakeFirst()
    )

    co2_emissions = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    particle_filter = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    revisions_book_complete = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    non_smoker = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    two_key = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    consumption_urban = Field(
        input_processor=MapCompose(do_strip, consumption_parser),
        output_processor=TakeFirst()
    )

    consumption_extra_urban = Field(
        input_processor=MapCompose(do_strip, consumption_parser),
        output_processor=TakeFirst()
    )

    consumption_combined = Field(
        input_processor=MapCompose(do_strip, consumption_parser),
        output_processor=TakeFirst()
    )

    open_ceiling = Field(
        input_processor=MapCompose(do_strip, ceiling_parser),
        output_processor=TakeFirst()
    )

    alloy_wheels = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    alloy_wheels_size = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    upholstery = Field(
        input_processor=MapCompose(do_strip, upholstery_parser),
        output_processor=TakeFirst()
    )

    airbags_n = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    air_conditioning = Field(
        input_processor=MapCompose(do_strip, aircon_parser),
        output_processor=TakeFirst()
    )

    vehicle_condition = Field(
        input_processor=MapCompose(do_strip, used_parser),
        output_processor=TakeFirst()
    )

    vin = Field(
        input_processor=MapCompose(do_strip),
        output_processor=TakeFirst()
    )

    accepts_recovery = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    mechancal_guaranty_until_date = Field(
        input_processor=MapCompose(do_strip, convert_date2),
        output_processor=TakeFirst()
    )

    mechancal_guaranty_until_mileage = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    registrations_n = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    registration_id = Field(
        input_processor=MapCompose(do_strip),
        output_processor=TakeFirst()
    )

    finance_possible = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    stand_guaranty_in_price = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    iuc = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    inspection_validity_date = Field(
        input_processor=MapCompose(do_strip, convert_date2),
        output_processor=TakeFirst()
    )

    electric_canopy = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    vat_deductable = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    price_without_iuc = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    canopy = Field(
        input_processor=MapCompose(do_strip, canopy_parser),
        output_processor=TakeFirst()
    )

    max_range = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    saved = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    classic = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    price_without_isv = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    stand_guaranty_not_in_price = Field(
        input_processor=MapCompose(do_strip, numeric_parser),
        output_processor=TakeFirst()
    )

    vat_discriminated = Field(
        input_processor=MapCompose(do_strip, sim_binary),
        output_processor=TakeFirst()
    )

    factory_guaranty_until_date = Field(
        input_processor=MapCompose(do_strip, convert_date2),
        output_processor=TakeFirst()
    )

    damaged = Field(
        input_processor=MapCompose(),
        output_processor=TakeFirst()
    )