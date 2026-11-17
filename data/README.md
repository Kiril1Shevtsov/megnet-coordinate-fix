# Исходные структуры

Скачайте [dichalcogenides_public.tar.gz из официального репозитория IDAO 2022](https://raw.githubusercontent.com/HSE-LAMBDA/IDAO-2022/3f1b62cf2a56449c3a02878137fce0af5714c0e3/data/dichalcogenides_public.tar.gz). SHA-256: 70deb90928bdb31a5db14addea5b18215b8eda1565f6ed42224f3f4ee57ef29b. Исходные структуры и целевые метки берутся у авторов, а не распространяются в этом репозитории.

~~~bash
python prepare_data.py /path/to/dichalcogenides_public.tar.gz
~~~

Скрипт проверяет хеш, восстанавливает 2966 объектов и создаёт локальный data/structures.jsonl.gz. Для каждого материала сохраняются ID, исходная цель, флаг изменения и два списка сайтов: baseline и corrected. Каждый сайт состоит из обозначения элемента и трёх дробных координат; флаг изменения истинен для 14 объектов.

[Постановка задачи IDAO](https://github.com/HSE-LAMBDA/IDAO-2022) · [Изучаемый предобработчик](https://github.com/yk4r2/idao_22/blob/956a3c44235d767dc4579b388433d084afb8aa44/adhoc/scripts/atoms_to_defects.py) · [MEGNet](https://github.com/materialsvirtuallab/megnet).
