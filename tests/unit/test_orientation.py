from PIL import Image

from npa_ocr.orientation import readability, confirm_rotation, check_osd_proposal


def reading(words, confidence):
    return readability(dict(text=words.split(), conf=[confidence]*len(words.split())))


def test_false_osd_flip_does_not_override_readable_original():
    original=reading('Наименование вещества фактическое значение требуемое значение',90)
    upside_down=reading('вв нп ку аа тт 0000 1111 8888',12)
    result=confirm_rotation(180,original,upside_down)
    assert result==0
    assert result.review_issue['reason']=='orientation_proposal_unconfirmed'
    assert result.review_issue['proposed_rotation']==180


def test_clear_improvement_confirms_sideways_table_rotation():
    before=reading('шш вв аа нн',15)
    after=reading('Наименование вещества фактическое значение требуемое значение',90)
    result=confirm_rotation(90,before,after)
    assert result==90 and result.review_issue is None


def test_numeric_cells_alone_cannot_confirm_a_rotation():
    before=reading('Наименование объекта содержание углерода массовая доля',85)
    numbers=reading('1000 8000 0000 9999 72.0 57.8',100)
    result=confirm_rotation(180,before,numbers)
    assert result==0


def test_short_side_heading_cannot_overrule_main_body():
    before=reading('Наименование вещества фактическое значение требуемое значение',90)
    after=reading('Приложение перечень технологий',95)
    assert confirm_rotation(90,before,after)==0


def test_confirmation_uses_both_orientations_without_mutating_source():
    image=Image.new('RGB',(1000,2000),'white')
    sizes=[]
    def read(sample):
        sizes.append(sample.size)
        words='шш вв аа' if sample.width<sample.height else (
            'Наименование вещества фактическое значение требуемое значение')
        return dict(text=words.split(),conf=[90]*len(words.split()))
    assert check_osd_proposal(image,90,read)==90
    assert sizes==[(1000,2000),(2000,1000)]
    assert image.size==(1000,2000)


def test_zero_proposal_keeps_source_without_extra_ocr():
    def forbidden(*args):raise AssertionError('No rotation was proposed')
    assert check_osd_proposal(Image.new('RGB',(10,10)),0,forbidden)==0


def test_tall_grid_can_recover_sideways_body_when_osd_follows_heading():
    image=Image.new('RGB',(1000,2000),'white')
    calls=[]
    def read(sample):
        calls.append(sample.size)
        words=('Наименование вещества фактическое значение требуемое значение'
               if len(calls)==2 else 'шш вв аа')
        return dict(text=words.split(),conf=[90]*len(words.split()))
    result=check_osd_proposal(image,0,read,table_box=(100,400,800,1800))
    assert result==90
    assert len(calls)==4
    assert calls[0]==(700,1400)


def test_frame_alone_does_not_trigger_table_rotation_search():
    from PIL import ImageDraw
    from npa_ocr.orientation import dominant_table_box
    image=Image.new('RGB',(1000,1500),'white')
    draw=ImageDraw.Draw(image)
    draw.rectangle((30,30,970,1470),outline='black',width=3)
    assert dominant_table_box(image) is None
    for x in (150,400,650,850):draw.line((x,200,x,1300),fill='black',width=3)
    for y in (200,500,800,1100,1300):draw.line((150,y,850,y),fill='black',width=3)
    box=dominant_table_box(image)
    assert box is not None
    assert box[2]-box[0] < 800
    assert box[3]-box[1] > 1000


def test_confident_short_fragments_cannot_establish_new_axis_when_osd_is_zero():
    calls=[]
    def read(sample):
        calls.append(True)
        words=('Наименование вещества фактическое значение требуемое значение'
               if len(calls)==1 else ' '.join(['CO','NO','Na','Cl']*20))
        return dict(text=words.split(),conf=[95]*len(words.split()))
    assert check_osd_proposal(Image.new('RGB',(1000,2000)),0,read,
                              table_box=(100,400,800,1800))==0
