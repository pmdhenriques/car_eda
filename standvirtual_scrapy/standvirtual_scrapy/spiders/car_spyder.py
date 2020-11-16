import scrapy
from scrapy.loader import ItemLoader
from standvirtual_scrapy.items import CarItem
import numpy as np

labels = [
    [
    'Anunciante', 'Marca', 'Modelo', 'Série', 'Versão', 'Cilindrada', 'Segmento', 'Cor', 'Metalizado', 'Tipo de Caixa',
    'Número de Mudanças', 'Nº de portas', 'Lotação', 'Classe do veículo', 'Tracção', 'Emissões CO2', 'Filtro de Particulas',
    'Livro de Revisões completo', 'Não fumador', '2º Chave', 'Consumo Urbano', 'Consumo Extra Urbano', 'Consumo Combinado', 
    'Tecto de Abrir', 'Jantes de Liga Leve', 'Medida Jantes de Liga Leve', 'Estofos', 'Numero de Airbags', 'Ar Condicionado',
    'Condição', 'VIN', 'Aceita retoma', 'Garantia mecanica fabricante até', 'ou até', 'Registo(s)', 'Matrícula',
    'Possibilidade de financiamento', 'Garantia de Stand (incl. no preço)', 'IUC', 'Inspecção válida até', 'Capota Eléctrica', 
    'IVA dedutível', 'Valor sem IUC', 'Capota', 'Autonomia Máxima', 'Salvado', 'Clássico', 'Valor sem ISV', 
    'Garantia de Stand (incl. no preço p/mutuo acordo)', 'IVA Discriminado', 'Garantia de Fábrica até'],
    ['advertiser', 'brand', 'model', 'series', 'version', 'cylinder', 'segment', 'color', 'metallic', 'gear_type', 'gears_n', 
    'doors_n', 'capacity', 'car_class', 'traction', 'co2_emissions', 'particle_filter', 'revisions_book_complete', 'non_smoker', 
    'two_key', 'consumption_urban', 'consumption_extra_urban', 'consumption_combined', 'open_ceiling', 'alloy_wheels', 
    'alloy_wheels_size', 'upholstery', 'airbags_n', 'air_conditioning', 'vehicle_condition', 'vin', 'accepts_recovery', 
    'mechancal_guaranty_until_date', 'mechancal_guaranty_until_mileage', 'registrations_n', 'registration_id', 'finance_possible', 
    'stand_guaranty_in_price', 'iuc', 'inspection_validity_date', 'electric_canopy', 'vat_deductable', 'price_without_iuc', 
    'canopy', 'max_range', 'saved', 'classic', 'price_without_isv', 'stand_guaranty_not_in_price', 'vat_discriminated', 
    'factory_guaranty_until_date']]

class CarSpider(scrapy.Spider):
    name = "standvirtual"

    start_urls = ['https://www.standvirtual.com/carros/usados/']

    def parse(self, response):

        cars = response.css("article .offer-item__content")
        for car in cars:
            loader = ItemLoader(item=CarItem(), selector=car)            
            loader.add_css('guaranty', '.tag::text')
            loader.add_css('title', '.offer-title__link::text')
            loader.add_css('fuel', 'li[data-code=fuel_type] span::text')
            loader.add_css('first_registration_month', 'li[data-code=first_registration_month] span::text')
            loader.add_css('first_registration_year', 'li[data-code=first_registration_year] span::text')
            loader.add_css('mileage', 'li[data-code=mileage] span::text')
            loader.add_css('power', 'li[data-code=power] span::text')
            loader.add_css('price', 'span.offer-price__number span::text')
            loader.add_css('negociable', 'span[data-type=price_negotiable]::text')
            loader.add_css('city', 'span.ds-location-city::text')
            loader.add_css('region', 'span.ds-location-region::text')
            loader.add_css('origin', 'li[data-code=origin]::text')
            loader.add_css('link', '.offer-item__title a::attr(href)')
            car_item = loader.load_item()

            car_url = car.css('.offer-item__title a::attr(href)').get()
            if car_url != None:
                yield response.follow(car_url, self.parse_car, meta={'car_item': car_item})

        for a in response.css('li.next a'):
            yield response.follow(a, callback=self.parse)

    def parse_car(self, response):
        car_item = response.meta['car_item']
        loader = ItemLoader(item=car_item, response=response)
        loader.add_css('price_detail', 'span.offer-price__details::text')
        loader.add_css('post_date', 'span.offer-meta__value::text')
            
        car_params = response.css('li.offer-params__item')
        for p in car_params:
            p_label = p.css('span.offer-params__label::text').get().strip()
            lix = [i for i, x in enumerate(labels[0]) if p_label == x]
            if len(lix) > 0:
                p_label = labels[1][lix[0]]
                p_value = p.css('div.offer-params__value a::text').get()
                if p_value == None:
                    p_value = p.css('div.offer-params__value::text').get()
                if p_value != None:
                    loader.add_value(p_label, p_value)

        car_extras = response.css(".offer-features__group h5::text").getall()
        if len(car_extras) > 0:
            damaged = int(any([x.lower() == 'sinistrado' for x in car_extras]))
        else:
            damaged = 0
        loader.add_value('damaged', damaged)

        yield loader.load_item()
        