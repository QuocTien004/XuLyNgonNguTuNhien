import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

class TextAugmentation:
    @staticmethod
    def augment_text(text, augmentation_types):
        result = text
        for augmentation_type in augmentation_types:
            if augmentation_type == "synonym":
                aug = naw.SynonymAug(aug_src='wordnet')
            elif augmentation_type == "swap":
                aug = naw.RandomWordAug(action="swap")
            elif augmentation_type == "delete":
                aug = naw.RandomWordAug(action="delete")
            elif augmentation_type == "noise":
                aug = nac.RandomCharAug(action="insert")
            else:
                continue
            
            augmented = aug.augment(result)
            result = " ".join(augmented) if isinstance(augmented, list) else augmented
        
        return result