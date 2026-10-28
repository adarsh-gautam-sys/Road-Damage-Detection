import { useMemo } from "react";
import { MapContainer, Marker, TileLayer, useMapEvents } from "react-leaflet";
import { pinIcon } from "../lib/leafletIcons";

export interface LatLng {
  lat: number;
  lng: number;
}

const INDIA_CENTER: LatLng = { lat: 22.5, lng: 79 };

function ClickToPlace({ onPick }: { onPick: (p: LatLng) => void }) {
  useMapEvents({
    click(e) {
      onPick({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });
  return null;
}

/** Small map to correct the location of a photo: tap to place, or drag the pin. */
export function LocationPicker({ value, onChange }: { value: LatLng | null; onChange: (p: LatLng) => void }) {
  const icon = useMemo(() => pinIcon(), []);
  const center = value ?? INDIA_CENTER;
  return (
    <div className="h-56 overflow-hidden rounded-lg border border-etch-strong">
      <MapContainer
        center={[center.lat, center.lng]}
        zoom={value ? 17 : 5}
        className="h-full w-full"
        scrollWheelZoom={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          maxZoom={19}
        />
        <ClickToPlace onPick={onChange} />
        {value && (
          <Marker
            position={[value.lat, value.lng]}
            icon={icon}
            draggable
            eventHandlers={{
              dragend(e) {
                const ll = (e.target as L.Marker).getLatLng();
                onChange({ lat: ll.lat, lng: ll.lng });
              },
            }}
          />
        )}
      </MapContainer>
    </div>
  );
}
